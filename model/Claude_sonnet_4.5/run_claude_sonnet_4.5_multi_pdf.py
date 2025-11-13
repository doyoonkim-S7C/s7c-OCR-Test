import boto3
from botocore.config import Config
import base64
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pdf2image import convert_from_path
import io
import os
from PyPDF2 import PdfReader, PdfWriter
from datetime import datetime
import argparse
import glob
from pathlib import Path
import time

def process_pdf_batch(pdf_path, start_page, end_page, batch_id):
    """각 배치를 처리하는 함수 - 이미지 변환 방식"""
    print(f"배치 {batch_id} 처리 시작: 페이지 {start_page+1}-{end_page}")

    # 시간 측정 시작
    batch_start_time = time.time()
    image_conversion_start = time.time()

    try:
        config = Config(read_timeout=1000)

        # AWS Bedrock 클라이언트 설정
        bedrock = boto3.client(
            service_name='bedrock-runtime',
            region_name='us-east-1',
            config=config,
        )

        # PDF를 이미지로 변환 (해당 페이지 범위만)
        print(f"🖼️  PDF를 이미지로 변환 중... (페이지 {start_page+1}-{end_page})")

        # pdf2image로 특정 페이지 범위를 이미지로 변환
        images = convert_from_path(
            pdf_path,
            dpi=200,  # 고품질 이미지를 위한 DPI 설정
            first_page=start_page + 1,  # pdf2image는 1-based indexing
            last_page=end_page,
            fmt='PNG'
        )

        # 이미지들을 base64로 인코딩하여 Claude에게 전송
        image_contents = []

        for i, image in enumerate(images):
            # 이미지를 바이트로 변환
            img_byte_array = io.BytesIO()
            image.save(img_byte_array, format='PNG')
            img_byte_array = img_byte_array.getvalue()

            # base64 인코딩
            image_b64 = base64.standard_b64encode(img_byte_array).decode('utf-8')

            # 이미지 컨텐츠 추가
            image_contents.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/png",
                    "data": image_b64
                }
            })

        # 이미지 변환 시간 측정
        image_conversion_time = time.time() - image_conversion_start
        print(f"✅ {len(images)}개 페이지를 이미지로 변환 완료 ({image_conversion_time:.2f}초)")

        # 추론 시작 시간 측정
        inference_start_time = time.time()

        # 메시지 컨텐츠 구성 (이미지들 + 텍스트 프롬프트)
        content = []

        # 모든 이미지 추가
        content.extend(image_contents)

        # 텍스트 프롬프트 추가
        content.append({
            "type": "text",
            "text": f"""Please convert all the content from these {len(images)} page images to markdown format.

IMPORTANT INSTRUCTIONS:
1. Process ALL content completely from all images - do not stop mid-way or use "##계속" or similar continuation markers
2. Extract ALL text, including headers, body text, captions, and footnotes from each image
3. Convert tables to HTML format with proper structure
4. Maintain the original formatting and structure as much as possible
5. Process the images in order and include ALL information from every image
6. Do not truncate or summarize - provide the complete content
7. If multiple images are provided, process them as consecutive pages

Please ensure you process the entire content from all images without any continuation markers or incomplete outputs."""
        })

        # AWS Bedrock Claude 모델 호출을 위한 메시지 구성
        body = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 32768,  # 최대 토큰 수 증가
            "messages": [
                {
                    "role": "user",
                    "content": content
                }
            ]
        }

        # AWS Bedrock을 통한 Claude 모델 호출
        response = bedrock.invoke_model(
            modelId='global.anthropic.claude-sonnet-4-5-20250929-v1:0',
            body=json.dumps(body),
            contentType='application/json'
        )

        # 응답 처리
        response_body = json.loads(response['body'].read())
        result = response_body['content'][0]['text']

        # 추론 시간 측정
        inference_time = time.time() - inference_start_time
        batch_total_time = time.time() - batch_start_time

        print(f"배치 {batch_id} 처리 완료! (추론 시간: {inference_time:.2f}초, 전체 시간: {batch_total_time:.2f}초)")
        return batch_id, start_page, end_page, result, inference_time, batch_total_time

    except Exception as e:
        print(f"배치 {batch_id} 처리 실패: {str(e)}")
        return batch_id, start_page, end_page, f"오류 발생: {str(e)}", 0.0, 0.0

def save_results_to_markdown(results, pdf_path, output_filename):
    """처리 결과를 마크다운 파일로 저장하는 함수"""
    try:
        with open(output_filename, 'w', encoding='utf-8') as f:
            # 각 배치 결과 작성 (타이밍 정보는 제외하고 내용만 저장)
            for result in results:
                batch_id, start_page, end_page, content = result[0], result[1], result[2], result[3]

                # 오류가 발생한 경우
                if content.startswith("오류 발생:"):
                    f.write(f"**⚠️ 처리 오류:** {content}\n\n")
                else:
                    # 정상 처리된 경우 내용 작성
                    f.write(f"{content}\n\n")

                f.write("---\n\n")

        print(f"✅ 마크다운 파일 저장 완료: {output_filename}")

    except Exception as e:
        print(f"❌ 마크다운 파일 저장 실패: {str(e)}")


def process_single_pdf(pdf_path, output_dir, pages_per_batch, max_workers):
    """단일 PDF 파일을 처리하는 함수"""
    print(f"\n📄 처리 중: {pdf_path}")

    # 전체 처리 시간 측정 시작
    pdf_start_time = time.time()

    # 출력 파일명 생성
    pdf_name = Path(pdf_path).stem
    output_filename = os.path.join(output_dir, f"{pdf_name}_claude_sonnet_4.5.mmd")

    # 배치 처리 실행
    results = process_pdf_with_batch_custom(pdf_path, output_filename, pages_per_batch, max_workers)

    # 전체 처리 시간 계산
    pdf_total_time = time.time() - pdf_start_time

    # 통계 계산
    if results:
        # 총 페이지 수 계산 (각 배치의 (end_page - start_page) 합계)
        total_pages = sum(result[2] - result[1] for result in results if len(result) > 5)

        # 순수 추론 시간 합계
        pure_inference_time = sum(result[4] for result in results if len(result) > 5 and isinstance(result[4], (int, float)))

        # 성능 정보 반환
        performance_info = {
            'input_file': pdf_path,
            'output_file': output_filename,
            'pure_inference_time': pure_inference_time,
            'total_processing_time': pdf_total_time,
            'num_pages': total_pages
        }

        return results, performance_info
    else:
        return results, None

def process_pdf_with_batch_custom(pdf_path, output_filename, pages_per_batch=5, max_workers=10):
    """배치 처리로 PDF 전체를 처리하는 메인 함수 (출력 파일명 지정 가능)"""

    # PDF 총 페이지 수 확인
    reader = PdfReader(pdf_path)
    total_pages = len(reader.pages)
    print(f"총 {total_pages}페이지 PDF 처리 시작")

    # 배치 생성
    batches = []
    for i in range(0, total_pages, pages_per_batch):
        start_page = i
        end_page = min(i + pages_per_batch, total_pages)
        batches.append((start_page, end_page))

    print(f"총 {len(batches)}개 배치로 분할")

    # 배치 처리 실행
    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 모든 배치 작업 제출
        future_to_batch = {
            executor.submit(process_pdf_batch, pdf_path, start, end, i): i
            for i, (start, end) in enumerate(batches)
        }

        # 완료된 작업들 수집
        for future in as_completed(future_to_batch):
            batch_id = future_to_batch[future]
            try:
                result = future.result()
                results.append(result)
            except Exception as exc:
                print(f"배치 {batch_id} 예외 발생: {exc}")

    # 결과를 페이지 순서대로 정렬
    results.sort(key=lambda x: x[1])  # start_page 기준으로 정렬

    # 마크다운 파일로 저장
    save_results_to_markdown(results, pdf_path, output_filename)

    print(f"\n✅ 처리 완료: {pdf_path}")
    print(f"결과 저장: {output_filename}")

    return results

def process_folder(input_dir, output_dir, pages_per_batch, max_workers):
    """폴더 내 모든 PDF 파일을 처리하는 함수 (하위 디렉토리 구조 유지)"""

    # 출력 디렉토리 생성
    os.makedirs(output_dir, exist_ok=True)

    # PDF 파일 찾기 (os.walk를 사용해서 하위 디렉토리까지 탐색)
    pdf_files = []
    print(f"📂 '{input_dir}' 디렉토리 및 하위 폴더에서 PDF 파일을 검색합니다...")

    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.lower().endswith('.pdf'):
                full_path = os.path.join(root, file)
                pdf_files.append(full_path)

    if not pdf_files:
        print(f"❌ {input_dir}에서 PDF 파일을 찾을 수 없습니다.")
        return []

    print(f"📂 총 {len(pdf_files)}개의 PDF 파일을 찾았습니다:")
    for pdf_file in pdf_files:
        # 상대 경로로 표시 (더 깔끔하게)
        rel_path = os.path.relpath(pdf_file, input_dir)
        print(f"  - {rel_path}")

    # 성능 로그 초기화
    performance_log = []

    # 각 PDF 파일 처리
    for i, pdf_file in enumerate(pdf_files, 1):
        # 상대 경로 계산 (원본 하위 폴더 구조 유지를 위해)
        rel_path = os.path.relpath(pdf_file, input_dir)

        print(f"\n{'='*60}")
        print(f"진행률: {i}/{len(pdf_files)} - {rel_path}")
        print(f"{'='*60}")

        try:
            # 출력 하위 디렉토리 경로 결정 (원본 구조 유지)
            output_subdir = os.path.join(output_dir, os.path.dirname(rel_path))

            # 하위 디렉토리 생성
            os.makedirs(output_subdir, exist_ok=True)

            results, performance_info = process_single_pdf(pdf_file, output_subdir, pages_per_batch, max_workers)

            if performance_info:
                performance_log.append(performance_info)

        except Exception as e:
            print(f"❌ {pdf_file} 처리 중 오류 발생: {str(e)}")
            continue

    print(f"\n🎉 모든 처리 완료! 총 {len(performance_log)}개 파일 처리됨")
    print(f"출력 디렉토리: {output_dir}")

    return performance_log

def generate_performance_report(performance_log, output_dir):
    """성능 보고서를 생성하고 저장하는 함수"""
    if not performance_log:
        print("성능 데이터가 없어 보고서를 생성할 수 없습니다.")
        return

    # 보고서 파일명 생성
    current_datetime = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_filename = f"performance_report_claude_sonnet_4.5_{current_datetime}.txt"
    report_path = os.path.join(output_dir, report_filename)

    with open(report_path, 'w', encoding='utf-8') as report_file:
        # 헤더 정보
        report_file.write("=" * 80 + "\n")
        report_file.write("Claude Sonnet 4.5 OCR 성능 보고서\n")
        report_file.write("=" * 80 + "\n")

        # 모델 정보
        report_file.write("모델 정보:\n")
        report_file.write("-" * 40 + "\n")
        report_file.write("모델: AWS Bedrock Claude Sonnet 4.5\n")
        report_file.write("모델 ID: global.anthropic.claude-sonnet-4.5-20250514-v1:0\n")
        report_file.write("처리 방식: PDF to Image + Batch Processing\n")
        report_file.write("이미지 포맷: PNG (200 DPI)\n")
        report_file.write("최대 토큰 수: 32768\n\n")

        # 전체 통계
        total_pdfs = len(performance_log)
        total_pages = sum(info['num_pages'] for info in performance_log)
        total_pure_inference_time = sum(info['pure_inference_time'] for info in performance_log)
        total_processing_time = sum(info['total_processing_time'] for info in performance_log)
        avg_inference_time_per_pdf = total_pure_inference_time / total_pdfs if total_pdfs > 0 else 0
        avg_inference_time_per_page = total_pure_inference_time / total_pages if total_pages > 0 else 0

        report_file.write("전체 통계:\n")
        report_file.write("-" * 40 + "\n")
        report_file.write(f"처리된 PDF 수: {total_pdfs}\n")
        report_file.write(f"총 페이지 수: {total_pages}\n")
        report_file.write(f"총 순수 추론 시간: {total_pure_inference_time:.2f}초\n")
        report_file.write(f"총 전체 처리 시간: {total_processing_time:.2f}초\n")
        report_file.write(f"PDF당 평균 추론 시간: {avg_inference_time_per_pdf:.2f}초\n")
        report_file.write(f"페이지당 평균 추론 시간: {avg_inference_time_per_page:.2f}초\n\n")

        # 개별 PDF 상세 정보
        report_file.write("개별 PDF 처리 결과:\n")
        report_file.write("=" * 80 + "\n")

        for i, info in enumerate(performance_log, 1):
            input_filename = os.path.basename(info['input_file'])
            output_filename = os.path.basename(info['output_file'])

            report_file.write(f"[{i:02d}] {input_filename}\n")
            report_file.write("-" * 60 + "\n")
            report_file.write(f"입력 파일: {input_filename}\n")
            report_file.write(f"출력 파일: {output_filename}\n")
            report_file.write(f"페이지 수: {info['num_pages']}\n")
            report_file.write(f"순수 추론 시간: {info['pure_inference_time']:.2f}초\n")
            report_file.write(f"전체 처리 시간: {info['total_processing_time']:.2f}초\n")

            if info['num_pages'] > 0:
                page_avg_inference = info['pure_inference_time'] / info['num_pages']
                report_file.write(f"페이지당 평균 추론 시간: {page_avg_inference:.2f}초\n")

            report_file.write("\n")

    print(f"✅ 성능 보고서가 저장되었습니다: {report_path}")
    return report_path

def main():
    parser = argparse.ArgumentParser(description="AWS Bedrock Claude를 사용한 PDF OCR 배치 처리")

    parser.add_argument(
        "-i", "--input",
        type=str,
        required=True,
        help="입력 PDF 파일 또는 폴더 경로"
    )

    parser.add_argument(
        "-o", "--output",
        type=str,
        required=True,
        help="출력 디렉토리 경로"
    )

    parser.add_argument(
        "--pages_per_batch",
        type=int,
        default=5,
        help="배치당 페이지 수 (기본값: 5)"
    )

    parser.add_argument(
        "--max_workers",
        type=int,
        default=2,
        help="최대 동시 처리 배치 수 (기본값: 2)"
    )

    parser.add_argument(
        "--region",
        type=str,
        default="us-east-1",
        help="AWS 리전 (기본값: us-east-1)"
    )

    args = parser.parse_args()

    # 스크립트 시작 시간 기록
    script_start_time = time.time()

    print("🚀 AWS Bedrock Claude OCR 배치 처리 시작")
    print(f"입력: {args.input}")
    print(f"출력: {args.output}")
    print(f"배치당 페이지 수: {args.pages_per_batch}")
    print(f"최대 동시 처리: {args.max_workers}")
    print(f"AWS 리전: {args.region}")
    print("-" * 60)

    # 성능 로그 초기화
    performance_log = []

    # 입력이 파일인지 폴더인지 확인
    if os.path.isfile(args.input):
        # 단일 파일 처리
        print("📄 단일 PDF 파일 처리 모드")
        os.makedirs(args.output, exist_ok=True)
        results, performance_info = process_single_pdf(args.input, args.output, args.pages_per_batch, args.max_workers)

        if performance_info:
            performance_log.append(performance_info)

    elif os.path.isdir(args.input):
        # 폴더 처리
        print("📂 폴더 처리 모드")
        performance_log = process_folder(args.input, args.output, args.pages_per_batch, args.max_workers)

    else:
        print(f"❌ 입력 경로가 존재하지 않습니다: {args.input}")
        return

    # 스크립트 종료 시간 계산
    script_end_time = time.time()
    total_script_time = script_end_time - script_start_time

    # 성능 보고서 생성
    if performance_log:
        generate_performance_report(performance_log, args.output)

        # 간단한 요약 출력
        total_pdfs = len(performance_log)
        total_pages = sum(info['num_pages'] for info in performance_log)
        total_inference_time = sum(info['pure_inference_time'] for info in performance_log)

        print(f"\n📊 처리 완료 요약:")
        print(f"   - 처리된 PDF: {total_pdfs}개")
        print(f"   - 총 페이지: {total_pages}페이지")
        print(f"   - 총 추론 시간: {total_inference_time:.2f}초")
        print(f"   - 전체 실행 시간: {total_script_time:.2f}초")
        if total_pages > 0:
            print(f"   - 페이지당 평균 추론 시간: {total_inference_time/total_pages:.2f}초")

    print(f"\n🎉 모든 작업이 완료되었습니다!")
    print(f"출력 디렉토리: {args.output}")

# 메인 실행
if __name__ == "__main__":
    main()