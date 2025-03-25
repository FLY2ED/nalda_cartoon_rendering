import cv2
import os
import numpy as np
from cartoon_cli import cartoonize_image

def main():
    # 예제 디렉토리 확인 및 생성
    examples_dir = "examples"
    os.makedirs(examples_dir, exist_ok=True)
    
    # 예제 이미지 디렉토리
    source_dir = "source_images"
    if not os.path.exists(source_dir):
        print(f"예제 이미지를 위한 디렉토리를 찾을 수 없습니다: {source_dir}")
        print("예제 이미지를 'source_images' 디렉토리에 넣어주세요.")
        return
    
    # 좋은 예제와 어려운 예제를 위한 이미지 파일 목록
    good_example = None
    challenging_example = None
    
    # 소스 디렉토리에서 이미지 찾기
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    for filename in os.listdir(source_dir):
        _, ext = os.path.splitext(filename)
        if ext.lower() in image_extensions:
            # 첫 번째 이미지를 좋은 예제로, 두 번째 이미지를 어려운 예제로 사용
            if good_example is None:
                good_example = filename
            elif challenging_example is None:
                challenging_example = filename
                break
    
    if good_example is None:
        print("예제 이미지를 찾을 수 없습니다.")
        return
    
    # 좋은 예제 처리
    good_img = cv2.imread(os.path.join(source_dir, good_example))
    if good_img is not None:
        print(f"좋은 예제 이미지 변환 중: {good_example}")
        
        # 기본 설정으로 변환
        good_cartoon = cartoonize_image(good_img)
        
        # 원본과 변환 결과를 나란히 표시
        h, w = good_img.shape[:2]
        good_comparison = np.zeros((h, w*2, 3), dtype=np.uint8)
        good_comparison[:, :w] = good_img
        good_comparison[:, w:] = good_cartoon
        
        # 이미지에 설명 추가
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(good_comparison, "원본", (10, 30), font, 1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(good_comparison, "만화 스타일", (w+10, 30), font, 1, (0, 255, 0), 2, cv2.LINE_AA)
        
        # 결과 저장
        cv2.imwrite(os.path.join(examples_dir, "good_example.jpg"), good_comparison)
        print(f"좋은 예제 저장됨: good_example.jpg")
    
    # 어려운 예제 처리
    if challenging_example:
        challenging_img = cv2.imread(os.path.join(source_dir, challenging_example))
        if challenging_img is not None:
            print(f"어려운 예제 이미지 변환 중: {challenging_example}")
            
            # 기본 설정으로 변환
            challenging_cartoon = cartoonize_image(challenging_img)
            
            # 원본과 변환 결과를 나란히 표시
            h, w = challenging_img.shape[:2]
            challenging_comparison = np.zeros((h, w*2, 3), dtype=np.uint8)
            challenging_comparison[:, :w] = challenging_img
            challenging_comparison[:, w:] = challenging_cartoon
            
            # 이미지에 설명 추가
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(challenging_comparison, "원본", (10, 30), font, 1, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(challenging_comparison, "만화 스타일", (w+10, 30), font, 1, (0, 255, 0), 2, cv2.LINE_AA)
            
            # 결과 저장
            cv2.imwrite(os.path.join(examples_dir, "challenging_example.jpg"), challenging_comparison)
            print(f"어려운 예제 저장됨: challenging_example.jpg")
    
    print("예제 이미지 생성 완료!")

if __name__ == "__main__":
    main() 