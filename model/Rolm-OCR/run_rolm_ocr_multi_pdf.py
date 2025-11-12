import os
import fitz
import io
import base64
from tqdm import tqdm
import torch
from concurrent.futures import ThreadPoolExecutor
import argparse
import time
import datetime

parser = argparse.ArgumentParser(description="RolmOCR PDF 처리 스크립트")
parser.add_argument("-i", "--input", type=str, help="입력 디렉토리")
parser.add_argument("-o", "--output", type=str, help="출력 디렉토리")
args, unknown = parser.parse_known_args()

# os.environ['VLLM_USE_V1'] = '0'
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = (
    "expandable_segments:True,"  # 단편화된 메모리를 이어붙일 수 있게 허용
    "max_split_size_mb:64,"  # 너무 큰 청크로 할당하지 않도록 제한 (기본 512MB)
    "garbage_collection_threshold:0.6"  # 캐시된 메모리를 적극적으로 반환
)

if torch.version.cuda == '11.8':
    os.environ["TRITON_PTXAS_PATH"] = "/usr/local/cuda-11.8/bin/ptxas"
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

from PIL import Image
from vllm import LLM, SamplingParams
from transformers import AutoProcessor

# [RolmOCR 수정] RolmOCR 모델명 지정
MODEL_NAME = "reducto/RolmOCR"
NUM_WORKERS = 64
MAX_CONCURRENCY = 100

# [RolmOCR 추가] RolmOCR용 프로세서 로드
print(f"{MODEL_NAME}의 프로세서를 로드합니다...")
processor = AutoProcessor.from_pretrained(MODEL_NAME, trust_remote_code=True)

print(f"{MODEL_NAME} 모델을 vLLM으로 로드합니다...")
llm = LLM(
    model=MODEL_NAME,
    block_size=256,
    enforce_eager=False,
    trust_remote_code=True,
    max_model_len=8192,
    swap_space=0,
    max_num_seqs=MAX_CONCURRENCY,
    tensor_parallel_size=1,
    gpu_memory_utilization=0.9,
    disable_mm_preprocessor_cache=True,
    dtype="auto"
)

# RolmOCR에 맞는 stop token id 설정
stop_token_ids = [processor.tokenizer.eos_token_id]

# im_end_id가 있는지 확인하고 안전하게 추가
if hasattr(processor.tokenizer, 'im_end_id'):
    stop_token_ids.append(processor.tokenizer.im_end_id)
else:
    print("Warning: tokenizer에 im_end_id가 없습니다. eos_token_id만 사용합니다.")

sampling_params = SamplingParams(
    temperature=0.0,
    max_tokens=8192,
    stop_token_ids=stop_token_ids
)


class Colors:
    RED = '\033[31m'
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    BLUE = '\033[34m'
    RESET = '\033[0m'


def pdf_to_images_high_quality(pdf_path, dpi=144, image_format="PNG"):
    """
    pdf2images
    """
    images = []

    try:
        pdf_document = fitz.open(pdf_path)
    except Exception as e:
        print(f"{Colors.RED}오류: {pdf_path} 파일을 열 수 없습니다. {e}{Colors.RESET}")
        return images

    zoom = dpi / 72.0
    matrix = fitz.Matrix(zoom, zoom)

    for page_num in range(pdf_document.page_count):
        page = pdf_document[page_num]

        pixmap = page.get_pixmap(matrix=matrix, alpha=False)
        Image.MAX_IMAGE_PIXELS = None

        if image_format.upper() == "PNG":
            img_data = pixmap.tobytes("png")
            img = Image.open(io.BytesIO(img_data))
        else:
            img_data = pixmap.tobytes("png")
            img = Image.open(io.BytesIO(img_data))
            if img.mode in ('RGBA', 'LA'):
                background = Image.new('RGB', img.size, (255, 255, 255))
                background.paste(img, mask=img.split()[-1] if img.mode == 'RGBA' else None)
                img = background

        images.append(img)

    pdf_document.close()
    return images


def process_single_image(image):
    """
    단일 이미지 처리:
    Processor로 이미지와 프롬프트를 함께 텍스트로 변환하여 vLLM에 전달 가능한 형태로 만듦.
    """
    try:
        # Processor가 이미지 입력을 지원하므로, text + image를 함께 처리
        chat_prompt = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": "Return the plain text representation of this document as if you were reading it naturally.\n"}
                ]
            }
        ]

        # processor가 chat 템플릿을 자동 적용할 수 있는지 확인
        prompt_text = processor.tokenizer.apply_chat_template(chat_prompt, tokenize=False, add_generation_prompt=True)
        return {"prompt": prompt_text}

    except Exception as e:
        print(f"{Colors.RED}이미지 처리 중 오류 발생: {e}{Colors.RESET}")
        return None


# [RolmOCR 수정] process_pdf 함수 대폭 수정
def process_pdf(pdf_input_path, output_dir, base_filename):
    """
    단일 PDF 파일을 받아 처리하고, 지정된 output_dir에 .md 파일로 결과를 저장합니다.
    """
    try:
        print(f"\n{Colors.BLUE}--- 시작: {pdf_input_path} ---{Colors.RESET}")

        start_time = time.time()

        # 1. 출력 디렉토리 생성
        os.makedirs(output_dir, exist_ok=True)

        # 2. PDF에서 이미지 로드
        print(f'{Colors.YELLOW}PDF 로딩 중...{Colors.RESET}')
        images = pdf_to_images_high_quality(pdf_input_path)

        if not images:
            print(f"{Colors.RED}페이지가 없거나 파일을 읽을 수 없습니다: {pdf_input_path}{Colors.RESET}")
            return None

        # 3. 이미지 전처리 (vLLM 입력 형식으로 변환)
        with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
            batch_inputs = list(tqdm(
                executor.map(process_single_image, images),
                total=len(images),
                desc=f"이미지 전처리: {base_filename}"
            ))

        # 4. LLM 생성 실행 (순수 추론 시간 측정)
        print(f"{Colors.GREEN}LLM 생성 실행 중...{Colors.RESET}")
        inference_start_time = time.time()
        outputs_list = llm.generate(
            batch_inputs,
            sampling_params=sampling_params
        )
        inference_end_time = time.time()

        pure_inference_time = inference_end_time - inference_start_time

        # 5. 출력 경로 정의 (단일 .md 파일)
        mode_suffix = "_rolmocr"
        md_path = os.path.join(output_dir, f'{base_filename}{mode_suffix}.md')

        all_page_contents = ''

        print("결과 후처리 중...")
        # 6. 결과 후처리 (단순 텍스트 취합)
        for i, output in enumerate(outputs_list):
            content = output.outputs[0].text

            page_separator = f'\n\n<--- Page {i + 1} End --->\n\n'
            all_page_contents += content + page_separator

        # 7. 파일 저장 (단일 .md 파일)
        with open(md_path, 'w', encoding='utf-8') as afile:
            afile.write(all_page_contents)

        end_time = time.time()
        total_processing_time = end_time - start_time

        print(f"{Colors.GREEN}--- 완료: {pdf_input_path} -> {md_path} ---{Colors.RESET}")

        # 성능 정보 반환 (출력 파일 목록 간소화)
        performance_info = {
            'input_file': pdf_input_path,
            'output_files': {
                'md': md_path,
                # 'mmd_det': mmd_det_path, # [RolmOCR 삭제]
                # 'pdf_layout': pdf_out_path # [RolmOCR 삭제]
            },
            'pure_inference_time': pure_inference_time,
            'total_processing_time': total_processing_time,
            'num_pages': len(images)
        }

        return performance_info

    except Exception as e:
        print(f"{Colors.RED}파일 처리 중 심각한 오류 발생 {pdf_input_path}: {e}{Colors.RESET}")
        return None


if __name__ == "__main__":

    # [RolmOCR 수정] args.mode 대신 모델명 직접 출력
    print(f"사용 모델: {MODEL_NAME}")

    BASE_INPUT_DIR = args.input
    BASE_OUTPUT_DIR = args.output

    performance_log = []
    script_start_time = time.time()

    pdf_files_to_process = []
    print(f"'{BASE_INPUT_DIR}' 디렉토리 및 하위 폴더에서 PDF 파일을 검색합니다...")

    for root, dirs, files in os.walk(BASE_INPUT_DIR):
        for file in files:
            if file.lower().endswith('.pdf'):
                full_path = os.path.join(root, file)
                pdf_files_to_process.append(full_path)

    print(f"{Colors.GREEN}총 {len(pdf_files_to_process)}개의 PDF 파일을 찾았습니다.{Colors.RESET}")

    if not pdf_files_to_process:
        print(f"처리할 PDF 파일이 없습니다.")
    else:
        for pdf_input_path in tqdm(pdf_files_to_process, desc="전체 PDF 처리 진행률"):
            rel_path = os.path.relpath(pdf_input_path, BASE_INPUT_DIR)
            output_subdir = os.path.join(BASE_OUTPUT_DIR, os.path.dirname(rel_path))
            base_filename = os.path.splitext(os.path.basename(pdf_input_path))[0]

            # [RolmOCR 수정] prompt 인자 불필요
            performance_info = process_pdf(pdf_input_path, output_subdir, base_filename)

            if performance_info:
                performance_log.append(performance_info)

        script_end_time = time.time()
        total_script_time = script_end_time - script_start_time

        # --- 5. 성능 보고서 작성 및 저장 ---
        report_filename = f"performance_report_rolmocr_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        report_path = os.path.join(BASE_OUTPUT_DIR, report_filename)

        with open(report_path, 'w', encoding='utf-8') as report_file:
            report_file.write("=" * 80 + "\n")
            report_file.write("RolmOCR 성능 보고서\n")  # [RolmOCR 수정]
            report_file.write("=" * 80 + "\n")

            # [RolmOCR 수정] 모델 정보 업데이트
            report_file.write("모델 정보:\n")
            report_file.write("-" * 40 + "\n")
            report_file.write(f"MODEL_NAME: {MODEL_NAME}\n")
            report_file.write(f"MAX_CONCURRENCY: {MAX_CONCURRENCY}\n")
            report_file.write(f"NUM_WORKERS: {NUM_WORKERS}\n")
            report_file.write("\n")

            if performance_log:
                total_pdfs = len(performance_log)
                total_pages = sum(info['num_pages'] for info in performance_log)
                total_pure_inference_time = sum(info['pure_inference_time'] for info in performance_log)
                avg_inference_time_per_pdf = total_pure_inference_time / total_pdfs
                avg_inference_time_per_page = total_pure_inference_time / total_pages if total_pages > 0 else 0

                report_file.write("전체 통계:\n")
                report_file.write("-" * 40 + "\n")
                report_file.write(f"처리된 PDF 수: {total_pdfs}\n")
                report_file.write(f"총 페이지 수: {total_pages}\n")
                report_file.write(f"총 순수 추론 시간: {total_pure_inference_time:.2f}초\n")
                report_file.write(f"PDF당 평균 추론 시간: {avg_inference_time_per_pdf:.2f}초\n")
                report_file.write(f"페이지당 평균 추론 시간: {avg_inference_time_per_page:.2f}초\n\n")

                report_file.write("개별 PDF 처리 결과:\n")
                report_file.write("=" * 80 + "\n")

                for i, info in enumerate(performance_log, 1):
                    input_filename = os.path.basename(info['input_file'])
                    # [RolmOCR 수정] 출력 파일 정보 간소화
                    md_filename = os.path.basename(info['output_files']['md'])

                    report_file.write(f"[{i:02d}] {input_filename}\n")
                    report_file.write("-" * 60 + "\n")
                    report_file.write(f"입력 파일: {input_filename}\n")
                    report_file.write(f"출력 파일 (MD): {md_filename}\n")
                    report_file.write(f"페이지 수: {info['num_pages']}\n")
                    report_file.write(f"순수 추론 시간: {info['pure_inference_time']:.2f}초\n")
                    report_file.write(f"전체 처리 시간: {info['total_processing_time']:.2f}초\n")
                    report_file.write(f"페이지당 평균 추론 시간: {info['pure_inference_time'] / info['num_pages']:.2f}초\n")
                    report_file.write("\n")

        print(f"\n{Colors.GREEN}성능 보고서가 저장되었습니다: {report_path}{Colors.RESET}")

    print(f"\n{Colors.GREEN}모든 작업이 완료되었습니다.{Colors.RESET}")