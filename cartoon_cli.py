import cv2
import numpy as np
import os
import argparse
import sys

def cartoonize_image(img, edge_threshold1=70, edge_threshold2=200,
                    bilateral_d=7, bilateral_color=150, bilateral_space=150,
                    median_ksize=5, adaptive_blocksize=9, adaptive_c=2,
                    color_quantization=12, line_size=7, blur_amount=5,
                    saturation_factor=1.2):
    """
    이미지를 고품질 만화 스타일로 변환합니다.
    
    Args:
        img: 입력 이미지 (OpenCV 형식)
        edge_threshold1: Canny 엣지 검출기의 첫 번째 임계값
        edge_threshold2: Canny 엣지 검출기의 두 번째 임계값
        bilateral_d: 양방향 필터의 필터 크기
        bilateral_color: 양방향 필터의 색상 시그마
        bilateral_space: 양방향 필터의 공간 시그마
        median_ksize: 미디언 블러 커널 크기
        adaptive_blocksize: 적응형 임계값의 블록 크기
        adaptive_c: 적응형 임계값의 C 값
        color_quantization: 색상 양자화 클러스터 수
        line_size: 엣지 선 두께
        blur_amount: 최종 블러 정도
        saturation_factor: 채도 증가 계수
        
    Returns:
        고품질 만화 스타일로 변환된 이미지
    """
    # 원본 이미지 복사
    img_copy = img.copy()
    
    # 그레이스케일 변환
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 노이즈 제거를 위한 미디언 블러
    gray_blur = cv2.medianBlur(gray, median_ksize)
    
    # 엣지 검출 (향상된 방식)
    if edge_threshold1 < edge_threshold2:
        # Canny 엣지 검출
        edges = cv2.Canny(gray_blur, edge_threshold1, edge_threshold2)
        # 엣지 두껍게 만들기
        kernel = np.ones((line_size, line_size), np.uint8)
        edges = cv2.dilate(edges, kernel)
        edges = cv2.bitwise_not(edges)
        
        # 이미지 스무딩 - 부드러운 선을 위해
        edges = cv2.GaussianBlur(edges, (blur_amount, blur_amount), 0)
    else:
        # 적응형 임계값 사용 - 두 번째 방식
        # 더 강한 윤곽선을 위해 더 작은 C 값 사용
        edges = cv2.adaptiveThreshold(gray_blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
                                     cv2.THRESH_BINARY, adaptive_blocksize, adaptive_c)
    
    # 색상 평활화 - 다단계 처리
    # 첫 번째 양방향 필터 - 강한 스무딩
    color1 = cv2.bilateralFilter(img, bilateral_d, bilateral_color, bilateral_space)
    
    # 두 번째 양방향 필터 - 미세 조정
    color2 = cv2.bilateralFilter(color1, bilateral_d//2, bilateral_color*2, bilateral_space*2)
    
    # HSV로 변환하여 채도 조정 (더 생생한 만화 색상)
    hsv = cv2.cvtColor(color2, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    s = np.clip(s * saturation_factor, 0, 255).astype(np.uint8)  # 채도 증가
    hsv_enhanced = cv2.merge([h, s, v])
    color_enhanced = cv2.cvtColor(hsv_enhanced, cv2.COLOR_HSV2BGR)
    
    # 색상 양자화 - K-means 클러스터링으로 색상 수 제한
    Z = color_enhanced.reshape((-1, 3))
    Z = np.float32(Z)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.001)  # 더 정확한 양자화
    K = color_quantization  # 더 많은 색상으로 좀더 자연스러운 효과
    _, labels, centers = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_PP_CENTERS)  # 더 좋은 초기화
    centers = np.uint8(centers)
    res = centers[labels.flatten()]
    quantized = res.reshape((color_enhanced.shape))
    
    # 엣지와 색상 결합 - 향상된 방식
    edges_3channel = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    
    # 색상에 가중치를 두고 결합하여 대비 높이기
    alpha = 0.85  # 색상 가중치
    beta = 1.0 - alpha  # 엣지 가중치
    cartoon = cv2.addWeighted(quantized, alpha, edges_3channel, beta, 0)
    
    # 약간의 선명도 개선 (언샵 마스킹)
    blur = cv2.GaussianBlur(cartoon, (0, 0), 3.0)
    cartoon = cv2.addWeighted(cartoon, 1.5, blur, -0.5, 0)
    
    return cartoon

def main():
    # 명령줄 인자 파싱
    parser = argparse.ArgumentParser(description='이미지를 만화 스타일로 변환합니다.')
    parser.add_argument('input', help='입력 이미지 파일 또는 디렉토리')
    parser.add_argument('--output', '-o', help='출력 디렉토리 (기본값: ./output)')
    parser.add_argument('--edge1', type=int, default=70, help='첫 번째 엣지 임계값 (기본값: 70)')
    parser.add_argument('--edge2', type=int, default=200, help='두 번째 엣지 임계값 (기본값: 200)')
    parser.add_argument('--bilateral-d', type=int, default=7, help='양방향 필터 크기 (기본값: 7)')
    parser.add_argument('--bilateral-color', type=int, default=150, help='양방향 필터 색상 시그마 (기본값: 150)')
    parser.add_argument('--bilateral-space', type=int, default=150, help='양방향 필터 공간 시그마 (기본값: 150)')
    parser.add_argument('--median', type=int, default=5, help='미디언 블러 크기 (기본값: 5)')
    parser.add_argument('--adaptive-block', type=int, default=9, help='적응형 임계값 블록 크기 (기본값: 9)')
    parser.add_argument('--adaptive-c', type=int, default=2, help='적응형 임계값 C 값 (기본값: 2)')
    parser.add_argument('--colors', type=int, default=12, help='색상 양자화 클러스터 수 (기본값: 12)')
    parser.add_argument('--preset', choices=['default', 'strong_edges'], 
                       help='프리셋 사용 (기본값: none)')
    
    args = parser.parse_args()
    
    # 출력 디렉토리 설정
    output_dir = args.output if args.output else "output"
    os.makedirs(output_dir, exist_ok=True)
    
    # 프리셋 적용
    if args.preset == 'strong_edges':
        args.edge1 = 50
        args.edge2 = 150
        args.bilateral_d = 5
        args.bilateral_color = 100
        args.bilateral_space = 100
        args.median = 5
        args.adaptive_block = 7
        args.adaptive_c = 2
    
    # 입력이 디렉토리인지 파일인지 확인
    if os.path.isdir(args.input):
        # 디렉토리 내 모든 이미지 처리
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        processed_count = 0
        
        for filename in os.listdir(args.input):
            _, ext = os.path.splitext(filename)
            if ext.lower() in image_extensions:
                input_path = os.path.join(args.input, filename)
                output_path = os.path.join(output_dir, f"{os.path.splitext(filename)[0]}_cartoon{ext}")
                
                img = cv2.imread(input_path)
                if img is None:
                    print(f"경고: {input_path} 파일을 읽을 수 없습니다. 건너뜁니다.")
                    continue
                
                print(f"변환 중: {filename}")
                cartoon = cartoonize_image(img, args.edge1, args.edge2, args.bilateral_d, 
                                         args.bilateral_color, args.bilateral_space,
                                         args.median, args.adaptive_block, args.adaptive_c,
                                         args.colors, line_size=7, blur_amount=5, 
                                         saturation_factor=1.2)
                
                cv2.imwrite(output_path, cartoon)
                processed_count += 1
                
        if processed_count > 0:
            print(f"{processed_count}개 이미지 변환 완료. 출력 디렉토리: {output_dir}")
        else:
            print(f"디렉토리에서 처리할 이미지를 찾을 수 없습니다: {args.input}")
    
    else:
        # 단일 파일 처리
        if not os.path.isfile(args.input):
            print(f"오류: 입력 파일이 존재하지 않습니다: {args.input}")
            return 1
        
        # 입력 파일 이름에서 출력 파일 이름 생성
        filename = os.path.basename(args.input)
        name, ext = os.path.splitext(filename)
        output_path = os.path.join(output_dir, f"{name}_cartoon{ext}")
        
        # 이미지 로드
        img = cv2.imread(args.input)
        if img is None:
            print(f"오류: 이미지를 로드할 수 없습니다: {args.input}")
            return 1
        
        # 이미지 변환
        print(f"변환 중: {filename}")
        cartoon = cartoonize_image(img, args.edge1, args.edge2, args.bilateral_d, 
                                 args.bilateral_color, args.bilateral_space,
                                 args.median, args.adaptive_block, args.adaptive_c,
                                 args.colors, line_size=7, blur_amount=5, 
                                 saturation_factor=1.2)
        
        # 결과 저장
        cv2.imwrite(output_path, cartoon)
        print(f"변환 완료. 출력 파일: {output_path}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 