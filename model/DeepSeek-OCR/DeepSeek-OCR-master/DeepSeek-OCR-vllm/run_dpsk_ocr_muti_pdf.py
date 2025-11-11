import os
import fitz
import img2pdf
import io
import re
from tqdm import tqdm
import torch
from concurrent.futures import ThreadPoolExecutor
import argparse

# Parse command line arguments BEFORE importing config
parser = argparse.ArgumentParser(description="Deepseek OCR PDF 처리 스크립트")
parser.add_argument(
    "--mode",
    type=str,
    default="Gundam",
    choices=['Tiny', 'Small', 'Base', 'Large', 'Gundam'],
    help="OCR 모드 선택 (기본값: Gundam)"
)
# Add other arguments that we'll parse later
parser.add_argument("-i", "--input", type=str, help="입력 디렉토리")
parser.add_argument("-o", "--output", type=str, help="출력 디렉토리")
args = parser.parse_known_args()

# Set environment variable for MODE before importing config
os.environ['DEEPSEEK_MODE'] = args.mode

if torch.version.cuda == '11.8':
    os.environ["TRITON_PTXAS_PATH"] = "/usr/local/cuda-11.8/bin/ptxas"
os.environ['VLLM_USE_V1'] = '0'
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

from config import MODEL_PATH, INPUT_PATH, OUTPUT_PATH, PROMPT, SKIP_REPEAT, MAX_CONCURRENCY, NUM_WORKERS, CROP_MODE

from PIL import Image, ImageDraw, ImageFont
import numpy as np
from deepseek_ocr import DeepseekOCRForCausalLM

from vllm.model_executor.models.registry import ModelRegistry

from vllm import LLM, SamplingParams
from process.ngram_norepeat import NoRepeatNGramLogitsProcessor
from process.image_process import DeepseekOCRProcessor

ModelRegistry.register_model("DeepseekOCRForCausalLM", DeepseekOCRForCausalLM)

llm = LLM(
    model=MODEL_PATH,
    hf_overrides={"architectures": ["DeepseekOCRForCausalLM"]},
    block_size=256,
    enforce_eager=False,
    trust_remote_code=True,
    max_model_len=8192,
    swap_space=0,
    max_num_seqs=MAX_CONCURRENCY,
    tensor_parallel_size=1,
    gpu_memory_utilization=0.9,
    disable_mm_preprocessor_cache=True
)

logits_processors = [NoRepeatNGramLogitsProcessor(ngram_size=20, window_size=50, whitelist_token_ids={128821, 128822})]  # window for fast；whitelist_token_ids: <td>,</td>

sampling_params = SamplingParams(
    temperature=0.0,
    max_tokens=8192,
    logits_processors=logits_processors,
    skip_special_tokens=False,
    include_stop_str_in_output=True,
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
        return images  # 빈 리스트 반환

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


def pil_to_pdf_img2pdf(pil_images, output_path):
    if not pil_images:
        return

    image_bytes_list = []

    for img in pil_images:
        if img.mode != 'RGB':
            img = img.convert('RGB')

        img_buffer = io.BytesIO()
        img.save(img_buffer, format='JPEG', quality=95)
        img_bytes = img_buffer.getvalue()
        image_bytes_list.append(img_bytes)

    try:
        pdf_bytes = img2pdf.convert(image_bytes_list)
        with open(output_path, "wb") as f:
            f.write(pdf_bytes)

    except Exception as e:
        print(f"error: {e}")


def re_match(text):
    pattern = r'(<\|ref\|>(.*?)<\|/ref\|><\|det\|>(.*?)<\|/det\|>)'
    matches = re.findall(pattern, text, re.DOTALL)

    mathes_image = []
    mathes_other = []
    for a_match in matches:
        if '<|ref|>image<|/ref|>' in a_match[0]:
            mathes_image.append(a_match[0])
        else:
            mathes_other.append(a_match[0])
    return matches, mathes_image, mathes_other


def extract_coordinates_and_label(ref_text, image_width, image_height):
    try:
        label_type = ref_text[1]
        cor_list = eval(ref_text[2])
    except Exception as e:
        print(e)
        return None

    return (label_type, cor_list)


def draw_bounding_boxes(image, refs, jdx, image_save_dir):  # <-- 인자 추가

    image_width, image_height = image.size
    img_draw = image.copy()
    draw = ImageDraw.Draw(img_draw)

    overlay = Image.new('RGBA', img_draw.size, (0, 0, 0, 0))
    draw2 = ImageDraw.Draw(overlay)

    #     except IOError:
    font = ImageFont.load_default()

    img_idx = 0

    for i, ref in enumerate(refs):
        try:
            result = extract_coordinates_and_label(ref, image_width, image_height)
            if result:
                label_type, points_list = result

                color = (np.random.randint(0, 200), np.random.randint(0, 200), np.random.randint(0, 255))

                color_a = color + (20,)
                for points in points_list:
                    x1, y1, x2, y2 = points

                    x1 = int(x1 / 999 * image_width)
                    y1 = int(y1 / 999 * image_height)

                    x2 = int(x2 / 999 * image_width)
                    y2 = int(y2 / 999 * image_height)

                    if label_type == 'image':
                        try:
                            cropped = image.crop((x1, y1, x2, y2))
                            # [수정됨] 하드코딩된 경로 대신 동적 경로 사용
                            cropped.save(os.path.join(image_save_dir, f"{jdx}_{img_idx}.jpg"))
                        except Exception as e:
                            print(e)
                            pass
                        img_idx += 1

                    try:
                        if label_type == 'title':
                            draw.rectangle([x1, y1, x2, y2], outline=color, width=4)
                            draw2.rectangle([x1, y1, x2, y2], fill=color_a, outline=(0, 0, 0, 0), width=1)
                        else:
                            draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
                            draw2.rectangle([x1, y1, x2, y2], fill=color_a, outline=(0, 0, 0, 0), width=1)

                        text_x = x1
                        text_y = max(0, y1 - 15)

                        text_bbox = draw.textbbox((0, 0), label_type, font=font)
                        text_width = text_bbox[2] - text_bbox[0]
                        text_height = text_bbox[3] - text_bbox[1]
                        draw.rectangle([text_x, text_y, text_x + text_width, text_y + text_height],
                                       fill=(255, 255, 255, 30))

                        draw.text((text_x, text_y), label_type, font=font, fill=color)
                    except:
                        pass
        except:
            continue
    img_draw.paste(overlay, (0, 0), overlay)
    return img_draw


# [수정] image_save_dir 인자 추가 및 전달
def process_image_with_refs(image, ref_texts, jdx, image_save_dir):
    result_image = draw_bounding_boxes(image, ref_texts, jdx, image_save_dir)
    return result_image


def process_single_image(image):
    """single image"""
    prompt_in = prompt
    cache_item = {
        "prompt": prompt_in,
        "multi_modal_data": {"image": DeepseekOCRProcessor().tokenize_with_images(images=[image], bos=True, eos=True,
                                                                                  cropping=CROP_MODE)},
    }
    return cache_item


# [새로 추가할 함수]
def process_pdf(pdf_input_path, output_dir, base_filename, prompt):
    """
    단일 PDF 파일을 받아 처리하고, 지정된 output_dir에 결과를 저장합니다.
    """
    try:
        print(f"\n{Colors.BLUE}--- 시작: {pdf_input_path} ---{Colors.RESET}")

        # 1. 출력 디렉토리 생성 (각 파일별로)
        os.makedirs(output_dir, exist_ok=True)
        image_output_dir = os.path.join(output_dir, 'images')
        os.makedirs(image_output_dir, exist_ok=True)

        # 2. PDF에서 이미지 로드
        print(f'{Colors.YELLOW}PDF 로딩 중...{Colors.RESET}')
        images = pdf_to_images_high_quality(pdf_input_path)

        if not images:
            print(f"{Colors.RED}페이지가 없거나 파일을 읽을 수 없습니다: {pdf_input_path}{Colors.RESET}")
            return

        # 3. 이미지 전처리
        with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
            batch_inputs = list(tqdm(
                executor.map(process_single_image, images),
                total=len(images),
                desc=f"이미지 전처리: {base_filename}"
            ))

        # 4. LLM 생성 실행
        print(f"{Colors.GREEN}LLM 생성 실행 중...{Colors.RESET}")
        outputs_list = llm.generate(
            batch_inputs,
            sampling_params=sampling_params
        )

        # 5. 출력 경로 정의 (상대 경로) - 모드 정보 포함
        mode_suffix = f"_{args.mode.lower()}"
        mmd_det_path = os.path.join(output_dir, f'{base_filename}_{mode_suffix}_det.mmd')
        mmd_path = os.path.join(output_dir, f'{base_filename}_{mode_suffix}.mmd')
        pdf_out_path = os.path.join(output_dir, f'{base_filename}_{mode_suffix}_layouts.pdf')

        contents_det = ''
        contents = ''
        draw_images = []
        jdx = 0

        print("결과 후처리 중...")
        # 6. 결과 후처리
        for output, img in zip(outputs_list, images):
            content = output.outputs[0].text

            if '<｜end▁of▁sentence｜>' in content:  # repeat no eos
                content = content.replace('<｜end▁of▁sentence｜>', '')
            else:
                if SKIP_REPEAT:
                    continue

            page_num = f'\n<--- Page Split --->'
            contents_det += content + f'\n{page_num}\n'
            image_draw = img.copy()

            matches_ref, matches_images, mathes_other = re_match(content)

            # [수정됨] image_output_dir 전달
            result_image = process_image_with_refs(image_draw, matches_ref, jdx, image_output_dir)

            draw_images.append(result_image)

            # mmd 파일 내의 이미지 경로가 'images/...'를 가리키도록 함
            for idx, a_match_image in enumerate(matches_images):
                content = content.replace(a_match_image, f'![](images/' + str(jdx) + '_' + str(idx) + '.jpg)\n')

            for idx, a_match_other in enumerate(mathes_other):
                content = content.replace(a_match_other, '').replace('\\coloneqq', ':=').replace('\\eqqcolon',
                                                                                                 '=:').replace(
                    '\n\n\n\n', '\n\n').replace('\n\n\n', '\n\n')

            contents += content + f'\n{page_num}\n'
            jdx += 1

        # 7. 파일 저장
        with open(mmd_det_path, 'w', encoding='utf-8') as afile:
            afile.write(contents_det)

        with open(mmd_path, 'w', encoding='utf-8') as afile:
            afile.write(contents)

        pil_to_pdf_img2pdf(draw_images, pdf_out_path)

        print(f"{Colors.GREEN}--- 완료: {pdf_input_path} -> {output_dir} ---{Colors.RESET}")

    except Exception as e:
        print(f"{Colors.RED}파일 처리 중 심각한 오류 발생 {pdf_input_path}: {e}{Colors.RESET}")


# [전체 교체]
if __name__ == "__main__":

    print(f"선택된 모드: {args.mode}")
    print(MODEL_PATH, INPUT_PATH, OUTPUT_PATH, PROMPT, SKIP_REPEAT, MAX_CONCURRENCY, NUM_WORKERS, CROP_MODE)

    # Use args from the top-level parsing
    # Set defaults if config values are empty
    final_input = args.input if args.input else INPUT_PATH
    final_output = args.output if args.output else OUTPUT_PATH

    # --- 2. 파싱된 인수를 변수에 할당 ---
    BASE_INPUT_DIR = final_input
    BASE_OUTPUT_DIR = final_output

    prompt = PROMPT  # 프롬프트는 config에서 그대로 사용

    # --- 3. 처리할 모든 PDF 파일 찾기 (os.walk) ---
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
        # --- 4. 각 PDF 파일 처리 (모델 로드는 1회만) ---
        for pdf_input_path in tqdm(pdf_files_to_process, desc="전체 PDF 처리 진행률"):
            # 1. 원본 하위 폴더 구조 유지를 위한 상대 경로 계산
            rel_path = os.path.relpath(pdf_input_path, BASE_INPUT_DIR)

            # 2. 출력 하위 디렉토리 경로 결정
            output_subdir = os.path.join(BASE_OUTPUT_DIR, os.path.dirname(rel_path))

            # 3. 저장할 파일 기본 이름 (확장자 제외)
            base_filename = os.path.splitext(os.path.basename(pdf_input_path))[0]

            # 4. 위에서 만든 process_pdf 함수 호출
            process_pdf(pdf_input_path, output_subdir, base_filename, prompt)

    print(f"\n{Colors.GREEN}모든 작업이 완료되었습니다.{Colors.RESET}")

