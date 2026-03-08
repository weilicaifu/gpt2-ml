#!/usr/bin/env python3
"""
CNKI (中国知网) 论文批量爬取程序
支持关键词搜索、批量获取论文元数据（标题、作者、摘要、关键词等）

使用方法:
  python cnki_scraper.py -k "机器学习" -n 100 -o results.csv
  python cnki_scraper.py -k "深度学习" -n 50 --start-year 2020 --end-year 2024 --format json
  python cnki_scraper.py -k "自然语言处理" --db journal --with-abstract -o nlp.csv
"""

import argparse
import csv
import json
import logging
import random
import re
import time
from dataclasses import asdict, dataclass, fields
from typing import List, Optional
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


@dataclass
class Paper:
    """论文元数据"""
    title: str = ""
    authors: str = ""
    source: str = ""         # 期刊/会议名称
    year: str = ""
    abstract: str = ""
    keywords: str = ""
    db_code: str = ""        # 数据库代码
    db_name: str = ""        # 数据库名称
    file_name: str = ""      # 文件ID
    cite_count: str = ""     # 被引次数
    download_count: str = "" # 下载次数
    url: str = ""


class CNKIScraper:
    """知网论文爬虫"""

    BASE_URL = "https://kns.cnki.net"
    SEARCH_URL = "https://kns.cnki.net/kns8/defaultresult/index"
    BRIEF_URL = "https://kns.cnki.net/kns8/Brief/GetGridTableHtml"
    DETAIL_URL = "https://kns.cnki.net/kns8/Detail/GetAbstract"

    # 各数据库代码
    DB_CODES = {
        "all":        "CJFD,CDFD,CMFD,CPFD,IPFD,CCND,BDZK",
        "journal":    "CJFD",   # 中国学术期刊
        "doctoral":   "CDFD",   # 博士论文
        "master":     "CMFD",   # 硕士论文
        "conference": "CPFD",   # 会议论文
    }

    HEADERS = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
        "Accept-Encoding": "gzip, deflate, br",
        "Connection": "keep-alive",
    }

    def __init__(self, delay_min: float = 1.5, delay_max: float = 4.0,
                 max_retries: int = 3, timeout: int = 20):
        self.delay_min = delay_min
        self.delay_max = delay_max
        self.max_retries = max_retries
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update(self.HEADERS)
        self._init_session()

    # ------------------------------------------------------------------
    # 会话初始化
    # ------------------------------------------------------------------

    def _init_session(self):
        """访问首页，获取必要的 Session Cookie"""
        try:
            resp = self.session.get(self.SEARCH_URL, timeout=self.timeout)
            resp.raise_for_status()
            logger.info("会话初始化成功，已获取 Cookies")
        except Exception as exc:
            logger.warning(f"会话初始化失败（可能影响爬取效果）: {exc}")

    # ------------------------------------------------------------------
    # 搜索入口
    # ------------------------------------------------------------------

    def search(
        self,
        keyword: str,
        max_results: int = 100,
        start_year: Optional[int] = None,
        end_year: Optional[int] = None,
        db_type: str = "all",
        with_abstract: bool = False,
    ) -> List[Paper]:
        """
        批量搜索论文

        Args:
            keyword:      搜索关键词
            max_results:  最大结果数
            start_year:   开始年份（含）
            end_year:     结束年份（含）
            db_type:      数据库类型，可选 all / journal / doctoral / master / conference
            with_abstract: 是否同时获取摘要（需要额外请求，速度较慢）

        Returns:
            Paper 列表
        """
        papers: List[Paper] = []
        page_size = 20
        page_num = 1

        logger.info(
            f"搜索开始 — 关键词: 「{keyword}」| 最多: {max_results} 篇 "
            f"| 数据库: {db_type}"
            + (f" | 年份: {start_year}–{end_year}" if start_year or end_year else "")
        )

        # 先访问搜索页面，触发服务端设置正确的 Cookie / 查询上下文
        self._visit_search_page(keyword, db_type)

        while len(papers) < max_results:
            batch = self._fetch_page_with_retry(
                keyword, page_num, page_size, start_year, end_year, db_type
            )

            if not batch:
                logger.info(f"第 {page_num} 页无结果，搜索结束")
                break

            papers.extend(batch)
            logger.info(
                f"已获取 {len(papers)} / {max_results} 篇（第 {page_num} 页，本页 {len(batch)} 篇）"
            )

            if len(batch) < page_size:
                logger.info("已到达最后一页")
                break

            page_num += 1
            self._sleep()

        papers = papers[:max_results]

        if with_abstract:
            self._enrich_abstracts(papers)

        return papers

    # ------------------------------------------------------------------
    # 内部：访问搜索页
    # ------------------------------------------------------------------

    def _visit_search_page(self, keyword: str, db_type: str):
        """访问搜索结果页面，令服务器写入查询 Cookie"""
        db_code = self.DB_CODES.get(db_type, self.DB_CODES["all"])
        params = {
            "DBCODE": db_code,
            "SFIELD": "SU",       # 主题
            "SKEY": keyword,
            "SORT": "relevance",
        }
        try:
            url = self.SEARCH_URL
            resp = self.session.get(url, params=params, timeout=self.timeout)
            resp.raise_for_status()
            self._sleep(0.5, 1.5)
        except Exception as exc:
            logger.warning(f"访问搜索页失败: {exc}")

    # ------------------------------------------------------------------
    # 内部：带重试的分页获取
    # ------------------------------------------------------------------

    def _fetch_page_with_retry(
        self, keyword, page_num, page_size, start_year, end_year, db_type
    ) -> List[Paper]:
        for attempt in range(1, self.max_retries + 1):
            try:
                return self._fetch_page(
                    keyword, page_num, page_size, start_year, end_year, db_type
                )
            except Exception as exc:
                wait = 2 ** attempt
                logger.warning(
                    f"第 {page_num} 页获取失败（第 {attempt} 次重试，等待 {wait}s）: {exc}"
                )
                time.sleep(wait)
        logger.error(f"第 {page_num} 页连续失败 {self.max_retries} 次，跳过")
        return []

    def _fetch_page(
        self, keyword, page_num, page_size, start_year, end_year, db_type
    ) -> List[Paper]:
        """向知网 Brief 接口发送 POST 请求，返回当页论文列表"""
        db_code = self.DB_CODES.get(db_type, self.DB_CODES["all"])

        query_json = self._build_query_json(keyword, start_year, end_year)

        data = {
            "IsSearch": "true",
            "QueryJson": json.dumps(query_json, ensure_ascii=False),
            "pageNum": str(page_num),
            "pageSize": str(page_size),
            "sortField": "publishdate",
            "sortType": "desc",
            "dstyle": "listmode",
            "boolSearch": "false",
            "dbPrefix": "SCDB",
            "dbCatalog": "中国学术文献网络出版总库",
            "ConfigFile": "SCDB.xml",
            "DisplayMode": "Lite",
            "searchFrom": "bn_left",
        }

        headers = {
            **self.HEADERS,
            "Referer": self.SEARCH_URL,
            "X-Requested-With": "XMLHttpRequest",
            "Content-Type": "application/x-www-form-urlencoded; charset=UTF-8",
        }

        resp = self.session.post(
            self.BRIEF_URL, data=data, headers=headers, timeout=self.timeout
        )
        resp.raise_for_status()
        return self._parse_brief_html(resp.text)

    # ------------------------------------------------------------------
    # 内部：构建查询 JSON
    # ------------------------------------------------------------------

    def _build_query_json(
        self, keyword: str, start_year: Optional[int], end_year: Optional[int]
    ) -> dict:
        q = {
            "Platform": "",
            "Resource": "TOTAL",
            "Classfilter": "",
            "Action": "Txt",
            "Content": keyword,
            "UKey": "",
            "NaviCode": "*",
            "ua": "1.21",
            "isinEn": "1",
            "SubjectValue": "",
            "SubjectLiteral": "",
            "SearchSystemValue": "",
            "SearchScopeName": "",
            "limitations": "",
            "ordertype": "3",
            "orderfields": "publishdate",
            "KuaKuCode": "CJFD,CDFD,CMFD,CPFD,IPFD,CCND,BDZK",
        }
        if start_year or end_year:
            yr = {}
            if start_year:
                yr["from"] = f"{start_year}0101"
            if end_year:
                yr["to"] = f"{end_year}1231"
            q["publishdate"] = yr
        return q

    # ------------------------------------------------------------------
    # 内部：解析搜索结果 HTML
    # ------------------------------------------------------------------

    def _parse_brief_html(self, html: str) -> List[Paper]:
        soup = BeautifulSoup(html, "html.parser")
        papers: List[Paper] = []

        # 结果行：<tr> 含 class="odd"/"even" 或直接所有 <tr>
        rows = soup.select("tr.odd, tr.even")
        if not rows:
            rows = soup.find_all("tr")

        for row in rows:
            paper = self._parse_row(row)
            if paper and paper.title:
                papers.append(paper)

        return papers

    def _parse_row(self, row) -> Optional[Paper]:
        """从一行 <tr> 中提取论文字段"""
        paper = Paper()

        # --- 标题 + URL ---
        title_cell = row.find("td", class_="name")
        if not title_cell:
            # 有些版本直接是 <a class="fz14">
            title_cell = row.find("a", class_="fz14")

        if not title_cell:
            return None

        link = title_cell.find("a") if title_cell.name != "a" else title_cell
        if not link:
            return None

        paper.title = link.get_text(strip=True)
        href = link.get("href", "")
        if href:
            paper.url = urljoin(self.BASE_URL, href) if href.startswith("/") else href
            # 从 URL 提取 dbcode / dbname / filename
            m = re.search(
                r"dbcode=([^&]+).*?dbname=([^&]+).*?filename=([^&]+)",
                href, re.IGNORECASE,
            )
            if m:
                paper.db_code = m.group(1)
                paper.db_name = m.group(2)
                paper.file_name = m.group(3)

        # --- 作者 ---
        author_td = row.find("td", class_="author")
        if author_td:
            paper.authors = author_td.get_text(strip=True)

        # --- 期刊/来源 ---
        source_td = row.find("td", class_="source")
        if source_td:
            paper.source = source_td.get_text(strip=True)

        # --- 年份 ---
        date_td = row.find("td", class_="date")
        if date_td:
            m = re.search(r"\d{4}", date_td.get_text())
            if m:
                paper.year = m.group()

        # --- 被引次数 ---
        quote_td = row.find("td", class_="quote")
        if quote_td:
            paper.cite_count = quote_td.get_text(strip=True)

        # --- 下载次数 ---
        dl_td = row.find("td", class_="download")
        if dl_td:
            paper.download_count = dl_td.get_text(strip=True)

        return paper

    # ------------------------------------------------------------------
    # 获取摘要
    # ------------------------------------------------------------------

    def _enrich_abstracts(self, papers: List[Paper]):
        """批量补充摘要字段"""
        logger.info(f"开始获取摘要，共 {len(papers)} 篇...")
        for i, paper in enumerate(papers, 1):
            if paper.abstract:
                continue
            abstract = self.fetch_abstract(paper)
            if abstract:
                paper.abstract = abstract
                logger.info(f"[{i}/{len(papers)}] 摘要获取成功: {paper.title[:40]}…")
            else:
                logger.debug(f"[{i}/{len(papers)}] 摘要为空: {paper.title[:40]}…")
            self._sleep()

    def fetch_abstract(self, paper: Paper) -> str:
        """获取单篇论文摘要"""
        if not paper.url:
            return ""
        try:
            resp = self.session.get(paper.url, timeout=self.timeout)
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, "html.parser")

            # 知网详情页摘要区域有多种 selector
            for selector in [
                "#ChDivSummary",
                ".abstract-text",
                "[id*='abstract']",
                ".abstract",
            ]:
                el = soup.select_one(selector)
                if el:
                    text = el.get_text(" ", strip=True)
                    # 去掉前缀"摘要："
                    text = re.sub(r"^摘\s*要[：:]?\s*", "", text)
                    if text:
                        return text

            # 关键词
            kw_el = soup.select_one(".keywords, #ChDivKeyWord")
            if kw_el and paper and not paper.keywords:
                paper.keywords = kw_el.get_text(" ", strip=True)

        except Exception as exc:
            logger.debug(f"摘要请求失败 ({paper.title[:20]}): {exc}")

        return ""

    # ------------------------------------------------------------------
    # 保存结果
    # ------------------------------------------------------------------

    def save_csv(self, papers: List[Paper], path: str):
        """保存为 CSV（UTF-8 BOM，可直接用 Excel 打开）"""
        if not papers:
            logger.warning("无数据可保存")
            return
        field_names = [f.name for f in fields(Paper)]
        with open(path, "w", newline="", encoding="utf-8-sig") as fh:
            writer = csv.DictWriter(fh, fieldnames=field_names)
            writer.writeheader()
            for p in papers:
                writer.writerow(asdict(p))
        logger.info(f"已保存 {len(papers)} 篇论文 → {path}")

    def save_json(self, papers: List[Paper], path: str):
        """保存为 JSON"""
        if not papers:
            logger.warning("无数据可保存")
            return
        with open(path, "w", encoding="utf-8") as fh:
            json.dump([asdict(p) for p in papers], fh, ensure_ascii=False, indent=2)
        logger.info(f"已保存 {len(papers)} 篇论文 → {path}")

    # ------------------------------------------------------------------
    # 工具方法
    # ------------------------------------------------------------------

    def _sleep(self, mn: Optional[float] = None, mx: Optional[float] = None):
        mn = mn if mn is not None else self.delay_min
        mx = mx if mx is not None else self.delay_max
        time.sleep(random.uniform(mn, mx))


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cnki_scraper",
        description="CNKI 中国知网论文批量爬取工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 搜索"机器学习"，获取 100 篇，保存为 CSV
  python cnki_scraper.py -k "机器学习" -n 100 -o ml_papers.csv

  # 搜索"深度学习"，限制 2020-2024 年，同时抓取摘要，输出 JSON
  python cnki_scraper.py -k "深度学习" -n 50 \\
      --start-year 2020 --end-year 2024 \\
      --with-abstract --format json -o dl_papers.json

  # 只搜索期刊论文
  python cnki_scraper.py -k "自然语言处理" --db journal -n 200 -o nlp.csv
        """,
    )
    parser.add_argument("-k", "--keyword", required=True, help="搜索关键词")
    parser.add_argument(
        "-n", "--num", type=int, default=100, metavar="N",
        help="最大爬取数量（默认 100）",
    )
    parser.add_argument(
        "-o", "--output", default="cnki_papers.csv", metavar="FILE",
        help="输出文件路径（默认 cnki_papers.csv）",
    )
    parser.add_argument(
        "--format", choices=["csv", "json"], default="csv",
        help="输出格式（默认 csv）",
    )
    parser.add_argument("--start-year", type=int, metavar="YEAR", help="开始年份（含）")
    parser.add_argument("--end-year",   type=int, metavar="YEAR", help="结束年份（含）")
    parser.add_argument(
        "--db",
        choices=["all", "journal", "doctoral", "master", "conference"],
        default="all",
        help="数据库范围（默认 all）",
    )
    parser.add_argument(
        "--with-abstract", action="store_true",
        help="同时抓取论文摘要（需额外请求，速度较慢）",
    )
    parser.add_argument(
        "--delay-min", type=float, default=1.5, metavar="SEC",
        help="最小请求间隔秒数（默认 1.5）",
    )
    parser.add_argument(
        "--delay-max", type=float, default=4.0, metavar="SEC",
        help="最大请求间隔秒数（默认 4.0）",
    )
    parser.add_argument(
        "--retries", type=int, default=3, metavar="N",
        help="失败重试次数（默认 3）",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="显示调试日志",
    )
    return parser


def resolve_output_path(path: str, fmt: str) -> str:
    """确保输出文件扩展名与格式一致"""
    ext = ".json" if fmt == "json" else ".csv"
    if not path.endswith(ext):
        base = path.rsplit(".", 1)[0] if "." in path else path
        return base + ext
    return path


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    scraper = CNKIScraper(
        delay_min=args.delay_min,
        delay_max=args.delay_max,
        max_retries=args.retries,
    )

    papers = scraper.search(
        keyword=args.keyword,
        max_results=args.num,
        start_year=args.start_year,
        end_year=args.end_year,
        db_type=args.db,
        with_abstract=args.with_abstract,
    )

    if not papers:
        print("未获取到任何结果，请检查网络连接或调整搜索参数")
        return

    output = resolve_output_path(args.output, args.format)

    if args.format == "json":
        scraper.save_json(papers, output)
    else:
        scraper.save_csv(papers, output)

    print(f"\n搜索完成！共获取 {len(papers)} 篇论文")
    print(f"结果已保存至: {output}")


if __name__ == "__main__":
    main()
