import argparse
import json
import os
import time
import datetime
from pathlib import Path
from mistralai import Mistral
from mistralai.client import MistralClient
from mistralai.models import OCRResponse
from mistralai import DocumentURLChunk
from IPython.display import Markdown  # (Jupyter/Colab 환경이 아니면 필요 X)

# --- 1. 명령줄 인수 파싱 함수 ---

def parse_arguments():
    """명령줄 인수를 파싱합니다."""
    parser = argparse.ArgumentParser(
        description="Mistral OCR을 사용하여 PDF 파일들을 마크다운으로 변환합니다.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
사용 예시:
  python run_mistral_ocr_multi_pdf.py --input /path/to/pdf/files --output /path/to/output
  python run_mistral_ocr_multi_pdf.py -i ./pdfs -o ./results
        """
    )

    parser.add_argument(
        "--input", "-i",
        type=str,
        required=True,
        help="처리할 PDF 파일들이 있는 입력 디렉터리 경로"
    )

    parser.add_argument(
        "--output", "-o",
        type=str,
        required=True,
        help="변환된 마크다운 파일들을 저장할 출력 디렉터리 경로"
    )

    parser.add_argument(
        "--api-key",
        type=str,
        help="Mistral API 키 (환경변수 MISTRAL_API_KEY로도 설정 가능)"
    )

    return parser.parse_args()


# --- 2. 마크다운 처리 함수 (먼저 정의) ---

def get_combined_markdown(ocr_response: OCRResponse) -> str:
    """모든 페이지의 마크다운을 하나로 합치고 이미지를 내장합니다."""
    markdowns: list[str] = []
    for page in ocr_response.pages:
        markdowns.append(page.markdown)

    return "\n\n".join(markdowns)


# --- 3. Mistral AI OCR 및 마크다운 변환 루프 ---
if __name__ == "__main__":
    # 명령줄 인수 파싱
    args = parse_arguments()

    # 입력 및 출력 디렉터리 설정
    input_dir = Path(args.input)
    output_dir = Path(args.output)

    # 입력 디렉터리 존재 확인
    if not input_dir.exists():
        print(f"❌ 오류: 입력 디렉터리가 존재하지 않습니다: {input_dir}")
        exit(1)

    # 출력 디렉터리 생성
    output_dir.mkdir(parents=True, exist_ok=True)

    # API 키 설정
    api_key = args.api_key or os.getenv("MISTRAL_API_KEY")
    if not api_key:
        print("❌ 오류: Mistral API 키가 필요합니다.")
        print("다음 중 하나의 방법으로 API 키를 제공하세요:")
        print("  1. --api-key 옵션 사용")
        print("  2. MISTRAL_API_KEY 환경변수 설정")
        exit(1)

    # Mistral 클라이언트 초기화
    try:
        client = Mistral(api_key=api_key)
        print("✅ Mistral 클라이언트가 성공적으로 초기화되었습니다.")
    except Exception as e:
        print(f"❌ 오류: Mistral 클라이언트 초기화 실패: {e}")
        exit(1)

    # --- 성능 보고서 초기화 ---
    performance_log = []
    current_datetime = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    print(f"'{input_dir}' 디렉터리 및 하위 폴더에서 PDF 파일을 검색합니다...")

    # .rglob("*.pdf")를 사용해 모든 하위 디렉터리의 PDF 파일을 재귀적으로 찾음
    pdf_files_to_process = list(input_dir.rglob("*.pdf"))

    if not pdf_files_to_process:
        print(f"처리할 PDF 파일을 찾을 수 없습니다: {input_dir}")
        exit(0)
    else:
        print(f"총 {len(pdf_files_to_process)}개의 PDF 파일을 찾았습니다.")

    # 각 PDF 파일에 대해 루프 실행
    for pdf_file in pdf_files_to_process:
        # 출력 파일 경로 생성
        output_mmd_file = output_dir / f"{pdf_file.stem}_Mistral-OCR.mmd"

        print(f"\n--- [시작] {pdf_file.name} 처리 중 ---")

        # 개별 파일 처리를 try...except로 감싸서 오류 발생 시 다음 파일로 넘어가도록 함
        try:
            print(f"업로드 중: {pdf_file.name}...")

            uploaded_file = client.files.upload(
                file={
                    "file_name": pdf_file.stem,
                    "content": pdf_file.read_bytes(),
                },
                purpose="ocr",
            )

            print("Signed URL 가져오는 중...")
            signed_url = client.files.get_signed_url(file_id=uploaded_file.id, expiry=1)

            print("OCR 처리 중...")
            ocr_start_time = time.time()

            pdf_response = client.ocr.process(
                document=DocumentURLChunk(document_url=signed_url.url),
                model="mistral-ocr-latest",
                include_image_base64=True
            )
            ocr_time = time.time() - ocr_start_time

            print("OCR 처리 완료.")

            # --- 4. 마크다운 생성 및 .mmd 파일로 저장 ---

            print("마크다운 생성 중...")

            # pdf_response 객체를 사용해 최종 마크다운 문자열 생성
            final_markdown_content = get_combined_markdown(pdf_response)

            # .mmd 파일로 저장 (UTF-8 인코딩 사용)
            with open(output_mmd_file, "w", encoding="utf-8") as f:
                f.write(final_markdown_content)

            print(f"✅ [성공] {pdf_file.name} -> {output_mmd_file.name} 파일로 저장되었습니다.")

            # 성능 정보 기록
            performance_info = {
                'input_file': str(pdf_file),
                'output_file': str(output_mmd_file),
                'num_pages': len(pdf_response.pages),
                'ocr_time': ocr_time,
            }
            performance_log.append(performance_info)

        except Exception as e:
            # 오류가 발생해도 다음 파일 처리를 계속함
            print(f"❌ [오류] {pdf_file.name} 처리 중 오류 발생: {e}")

            performance_info = {
                'input_file': str(pdf_file),
                'output_file': 'FAILED',
                'num_pages': 0,
                'ocr_time': 0,
                'error': str(e)
            }
            performance_log.append(performance_info)

    # --- 성능 보고서 작성 및 저장 ---
    report_filename = f"mistral_ocr_performance_report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    report_path = output_dir / report_filename

    with open(report_path, 'w', encoding='utf-8') as report_file:
        # 헤더 정보
        report_file.write("=" * 80 + "\n")
        report_file.write("Mistral OCR 성능 보고서\n")
        report_file.write("=" * 80 + "\n")
        report_file.write(f"실행 시간: {current_datetime}\n")
        report_file.write(f"입력 디렉터리: {input_dir}\n")
        report_file.write(f"출력 디렉터리: {output_dir}\n")
        report_file.write(f"모델: mistral-ocr-latest\n\n")

        # 전체 통계
        if performance_log:
            successful_files = [info for info in performance_log if 'error' not in info]
            failed_files = [info for info in performance_log if 'error' in info]

            total_pdfs = len(performance_log)
            successful_pdfs = len(successful_files)
            failed_pdfs = len(failed_files)

            if successful_files:
                total_pages = sum(info['num_pages'] for info in successful_files)
                total_ocr_time = sum(info['ocr_time'] for info in successful_files)
                avg_ocr_time_per_pdf = total_ocr_time / successful_pdfs if successful_pdfs > 0 else 0
                avg_ocr_time_per_page = total_ocr_time / total_pages if total_pages > 0 else 0

                report_file.write("전체 통계:\n")
                report_file.write("-" * 40 + "\n")
                report_file.write(f"처리된 PDF 수: {total_pdfs} (성공: {successful_pdfs}, 실패: {failed_pdfs})\n")
                report_file.write(f"총 페이지 수: {total_pages}\n")
                report_file.write(f"총 OCR 처리 시간: {total_ocr_time:.2f}초\n")
                report_file.write(f"PDF당 평균 OCR 시간: {avg_ocr_time_per_pdf:.2f}초\n")
                report_file.write(f"페이지당 평균 OCR 시간: {avg_ocr_time_per_page:.2f}초\n\n")

            # 개별 PDF 상세 정보
            report_file.write("개별 PDF 처리 결과:\n")
            report_file.write("=" * 80 + "\n")

            for i, info in enumerate(performance_log, 1):
                input_filename = Path(info['input_file']).name
                output_filename = Path(info['output_file']).name if info['output_file'] != 'FAILED' else 'FAILED'

                report_file.write(f"[{i:02d}] {input_filename}\n")
                report_file.write("-" * 60 + "\n")
                report_file.write(f"입력 파일: {input_filename}\n")
                report_file.write(f"출력 파일: {output_filename}\n")

                if 'error' in info:
                    report_file.write(f"상태: 실패\n")
                    report_file.write(f"오류 메시지: {info['error']}\n")
                else:
                    report_file.write(f"상태: 성공\n")
                    report_file.write(f"페이지 수: {info['num_pages']}\n")
                    report_file.write(f"OCR 처리 시간: {info['ocr_time']:.2f}초\n")
                    if info['num_pages'] > 0:
                        report_file.write(f"페이지당 평균 OCR 시간: {info['ocr_time']/info['num_pages']:.2f}초\n")

                report_file.write("\n")

    print(f"\n✅ 성능 보고서가 저장되었습니다: {report_path}")
    print(f"\n--- 모든 파일 처리 완료 ---")