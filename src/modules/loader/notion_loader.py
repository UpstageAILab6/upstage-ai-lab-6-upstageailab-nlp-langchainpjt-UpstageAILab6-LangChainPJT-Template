import json
import os
import re
import time
from typing import List, Optional

from langchain_text_splitters import RecursiveCharacterTextSplitter
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from langchain_core.documents import Document

from src.modules.loader.docs_loader import DocsLoader

class LectureLoader(DocsLoader):

    def load(self, source: str) -> List[Document]:
        pass


class LawLoader(DocsLoader):
    def load(self, source: str) -> List[Document]:
        # LawLoader.py 기준 두 단계 상위 디렉터리의 'files' 폴더에 있는 파일 경로
        file_path = os.path.join(
            os.path.dirname(__file__),  # 현재 파일이 있는 디렉터리
            "..",  # 한 단계 상위 디렉터리
            "..",  # 두 단계 상위 디렉터리
            "files",  # 상위 디렉터리 아래 'files' 폴더
            source  # 실제 파일 이름
        )

        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        chunk_size = 1000  # 원하는 chunk 길이
        chunk_overlap = 300  # 앞뒤로 겹칠 부분

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            # separators=["제\d+조\()"],
            # pattern=r'(?=제\d+조\()',  # lookahead 정규표현식
        )
        #
        documents = splitter.split_documents([Document(page_content=content)])

        return documents

class NotionLoader(DocsLoader):
    def __init__(self, driver: Optional[webdriver.Chrome] = None):
        # driver를 외부에서 주입받거나, read()에서 새롭게 생성합니다.
        self.driver = driver
        self.file_dir = os.path.join(os.path.dirname(__file__), "..", "..", "files")

    def load(self, source: str) -> List[Document]:
        """
        노션 페이지를 크롤링하여 HTML 내용과 다운로드 파일들을 Document 객체로 반환합니다.
        """
        chrome_options = Options()
        # 다운로드 폴더를 './files'로 지정하는 옵션 설정
        prefs = {
            "download.default_directory": self.file_dir,
            "download.prompt_for_download": False,
            "download.directory_upgrade": True,
            "safebrowsing.enabled": True
        }
        chrome_options.add_experimental_option("prefs", prefs)
        # 필요에 따라 헤드리스 모드 사용 가능
        chrome_options.add_argument("--headless=new")

        # driver가 없는 경우 새롭게 생성
        if self.driver is None:
            self.driver = webdriver.Chrome(options=chrome_options)
        else:
            # 외부에서 전달받은 driver에 옵션 적용은 별도 처리 필요
            self.driver.get(source)

        self.driver.get(source)
        time.sleep(1)

        # 페이지의 모든 콘텐츠 로드를 위해 스크롤
        self.scroll_to_bottom()

        # 노션 토글 블록 클릭 처리 (접힌 블록 펼치기)
        toggle_blocks = self.driver.find_elements(By.XPATH, "//div[contains(@class, 'notion-toggle-block')]")
        print(f"찾은 토글 블록 개수: {len(toggle_blocks)}")
        for toggle_block in toggle_blocks:
            try:
                self.driver.execute_script("arguments[0].scrollIntoView(true);", toggle_block)
                toggle_button = toggle_block.find_element(By.XPATH, ".//div[@role='button']")
                aria_expanded = toggle_button.get_attribute("aria-expanded")
                if aria_expanded == "false":
                    WebDriverWait(self.driver, 10).until(
                        lambda d: toggle_button.is_displayed() and toggle_button.is_enabled()
                    )
                    toggle_button.click()
                else:
                    print("이미 열린 토글:", toggle_block.text)
            except Exception as e:
                print("토글 블록 클릭 실패:", e)

        self.scroll_to_bottom()

        # .docx 파일 다운로드 처리 후, 다운로드된 파일 목록 반환
        downloaded_files = self.download_docx_files()

        # 페이지 HTML 저장 및 읽기
        page_html = self.driver.page_source
        html_file_path = os.path.join(self.file_dir, "page.html")
        with open(html_file_path, "w", encoding="utf-8") as html_file:
            html_file.write(page_html)
        # print(f"HTML 파일 저장됨: {html_file_path}")

        # driver 종료
        self.driver.quit()

        # Document 객체 생성: content에는 페이지 HTML, source에는 URL, attached_file에는 다운로드 파일 + HTML 파일
        attached_files = downloaded_files.copy()
        attached_files.append(html_file_path)
        attached_files_str = json.dumps(attached_files, ensure_ascii=False)

        return [Document(page_content=page_html,
                        metadata={"source": source, "attached_file": attached_files_str, "document_type": "html"})]

    def scroll_to_bottom(self, pause_time: int = 2) -> None:
        """
        페이지를 끝까지 스크롤하여 동적 로딩된 콘텐츠를 모두 표시하도록 합니다.
        """
        last_height = self.driver.execute_script("return document.body.scrollHeight")
        while True:
            self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(pause_time)
            new_height = self.driver.execute_script("return document.body.scrollHeight")
            if new_height == last_height:
                break
            last_height = new_height

    def wait_for_download_and_rename(self, download_folder: str, target_file_name: str, timeout: int = 30) -> None:
        """
        download_folder에서 .docx 파일이 다운로드 완료될 때까지 대기 후,
        가장 최근에 다운로드된 .docx 파일을 target_file_name으로 rename합니다.
        """
        end_time = time.time() + timeout
        downloaded_file = None
        while time.time() < end_time:
            files = [f for f in os.listdir(download_folder) if f.endswith(".docx") and not f.endswith(".crdownload")]
            if files:
                downloaded_file = max(files, key=lambda f: os.path.getmtime(os.path.join(download_folder, f)))
                break
            time.sleep(1)
        if downloaded_file:
            original_path = os.path.join(download_folder, downloaded_file)
            target_path = os.path.join(download_folder, target_file_name)
            if original_path != target_path:
                os.rename(original_path, target_path)
                # print(f"파일 이름 변경 완료: {downloaded_file} -> {target_file_name}")
        else:
            print("다운로드된 파일을 찾을 수 없습니다.")

    def download_docx_files(self, download_folder=None) -> List[str]:
        """
        노션 페이지 내 .docx 파일 블록을 클릭하여 파일 다운로드를 수행하고,
        다운로드된 파일 경로 목록을 반환합니다.
        """
        # self.file_dir
        if download_folder is None:
            download_folder = self.file_dir
        # if not os.path.exists(download_folder):
        #     os.makedirs(download_folder)
        docx_files = []
        file_blocks = self.driver.find_elements(By.XPATH, "//div[contains(@class, 'notion-file-block')]")
        for block in file_blocks:
            try:
                try:
                    file_name_element = block.find_element(By.XPATH, ".//*[contains(text(), '.docx')]")
                except Exception as e:
                    print("파일 이름 요소를 찾지 못했습니다:", e)
                    continue
                file_name = file_name_element.text.strip()
                # 파일명 내 공백을 '+'로 대체
                file_name = file_name.replace(" ", "+")
                download_path = os.path.abspath(download_folder)
                file_path = os.path.join(download_path, file_name)
                # print(f"다운로드 파일 경로: {file_path}")
                # 기본 파일명 비교 (숫자 접미사 무시)
                base_file_name = re.sub(r' \(\d+\)', '', file_name)
                existing_files = os.listdir(download_path)
                skip = False
                for existing in existing_files:
                    if existing.endswith('.docx'):
                        existing_base = re.sub(r' \(\d+\)', '', existing)
                        if existing_base == base_file_name:
                            skip = True
                            break
                if skip:
                    docx_files.append(os.path.join(download_path, file_name))
                    # print(f"{file_name} 이미 존재하여 다운로드 스킵.")
                    continue

                if ".docx" in file_name:
                    # print("다운로드 대상 파일:", file_name)
                    original_handle = self.driver.current_window_handle
                    block.click()
                    time.sleep(1)  # 새 탭 열림 대기

                    handles = self.driver.window_handles
                    if len(handles) > 1:
                        for handle in handles:
                            if handle != original_handle:
                                try:
                                    self.driver.switch_to.window(handle)
                                    # print("새 탭에서 자동 다운로드 진행중...")
                                    time.sleep(2)
                                    self.driver.close()
                                except Exception as e:
                                    print("새 탭 처리 중 오류:", e)
                        try:
                            self.driver.switch_to.window(original_handle)
                        except Exception as e:
                            print("원래 창으로 전환 실패:", e)
                    else:
                        print("새 탭이 열리지 않음, 클릭 후 다운로드가 진행되었을 가능성이 있음.")
                    time.sleep(1)
                    self.wait_for_download_and_rename(download_path, file_name)
                    docx_files.append(os.path.join(download_path, file_name))
            except Exception as e:
                print("파일 다운로드 처리 실패:", e)
        for file in docx_files:
            print(f"{file} 파일 다운로드 요청 완료.")
        print(f"다운로드된 파일 개수: {len(docx_files)}")
        # print(docx_files)
        return docx_files
