import cv2
import numpy as np
import os
import tkinter as tk
from tkinter import ttk, filedialog, font
from PIL import Image, ImageTk

class NALDACartoonRenderer:
    def __init__(self, root):
        self.root = root
        self.root.title("NALDA 만화 렌더링")
        self.root.geometry("1280x720")
        self.root.configure(bg="#f8f9fa")
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        # 이미지 변수
        self.original_image = None
        self.cartoon_image = None
        self.current_display = None
        self.filename = None
        
        # 출력 디렉토리 생성
        self.output_dir = "output"
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 예제 디렉토리 생성
        self.examples_dir = "examples"
        os.makedirs(self.examples_dir, exist_ok=True)
        
        # 변환 설정
        self.edge_threshold1 = 15
        self.edge_threshold2 = 30
        self.bilateral_d = 9
        self.bilateral_color = 250
        self.bilateral_space = 250
        self.median_ksize = 5
        self.adaptive_blocksize = 9
        self.adaptive_c = 5
        
        # 토스 스타일 색상
        self.colors = {
            'primary': "#3182f6",  # 토스 파란색
            'secondary': "#68aaff",
            'background': "#ffffff",
            'panel_bg': "#f5f6f7",
            'text': "#333333",
            'text_light': "#787878",
            'success': "#4cd964",
            'danger': "#fc3d39",
            'warning': "#ffb800",
            'border': "#eaeaea"
        }
        
        # 폰트 설정
        self.default_font = font.nametofont("TkDefaultFont")
        self.default_font.configure(family="맑은 고딕", size=10)
        self.title_font = font.Font(family="맑은 고딕", size=14, weight="bold")
        self.subtitle_font = font.Font(family="맑은 고딕", size=12, weight="bold")
        
        # UI 초기화
        self.setup_ui()
    
    def setup_ui(self):
        # 메인 프레임
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # 스타일 설정
        style = ttk.Style()
        style.configure("TFrame", background=self.colors['background'])
        style.configure("TLabel", background=self.colors['background'], font=self.default_font)
        style.configure("TButton", font=self.default_font)
        
        # 상단 프레임
        top_frame = ttk.Frame(main_frame)
        top_frame.pack(fill=tk.X, pady=(0, 10))
        
        # 앱 제목
        title_label = ttk.Label(top_frame, text="NALDA 만화 렌더링", font=self.title_font, foreground=self.colors['primary'])
        title_label.pack(side=tk.LEFT, padx=10)
        
        # 상태 표시
        self.status_label = ttk.Label(top_frame, text="이미지를 로드하세요", font=self.default_font, foreground=self.colors['text'])
        self.status_label.pack(side=tk.RIGHT, padx=10)
        
        # 중앙 프레임 (이미지 + 컨트롤)
        center_frame = ttk.Frame(main_frame)
        center_frame.pack(fill=tk.BOTH, expand=True)
        
        # 이미지 표시 영역
        image_frame = ttk.Frame(center_frame, relief=tk.GROOVE, borderwidth=2)
        image_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
        
        self.image_label = ttk.Label(image_frame)
        self.image_label.pack(fill=tk.BOTH, expand=True)
        
        # 오른쪽 컨트롤 패널
        controls_frame = ttk.Frame(center_frame, relief=tk.GROOVE, borderwidth=2, width=300)
        controls_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=(0, 0), ipadx=10, ipady=10)
        controls_frame.pack_propagate(False)  # 프레임 크기 고정
        
        # 파일 버튼
        file_frame = ttk.Frame(controls_frame)
        file_frame.pack(fill=tk.X, padx=10, pady=10)
        
        open_button = ttk.Button(file_frame, text="이미지 열기", command=self.open_image)
        open_button.pack(side=tk.LEFT, padx=(0, 5), fill=tk.X, expand=True)
        
        save_button = ttk.Button(file_frame, text="결과 저장", command=self.save_image)
        save_button.pack(side=tk.RIGHT, padx=(5, 0), fill=tk.X, expand=True)
        
        # 변환 버튼
        convert_button = ttk.Button(controls_frame, text="만화 스타일로 변환", command=self.convert_to_cartoon)
        convert_button.pack(fill=tk.X, padx=10, pady=5)
        
        # 원본/결과 전환 버튼
        toggle_frame = ttk.Frame(controls_frame)
        toggle_frame.pack(fill=tk.X, padx=10, pady=5)
        
        original_button = ttk.Button(toggle_frame, text="원본 보기", command=lambda: self.show_image("original"))
        original_button.pack(side=tk.LEFT, padx=(0, 5), fill=tk.X, expand=True)
        
        result_button = ttk.Button(toggle_frame, text="결과 보기", command=lambda: self.show_image("cartoon"))
        result_button.pack(side=tk.RIGHT, padx=(5, 0), fill=tk.X, expand=True)
        
        # 프리셋 선택 영역
        preset_label = ttk.Label(controls_frame, text="만화 스타일 선택", font=self.subtitle_font, foreground=self.colors['primary'])
        preset_label.pack(anchor=tk.W, padx=10, pady=(20, 10))
        
        # 프리셋 설명
        preset_desc = ttk.Label(controls_frame, text="스타일을 선택하여 이미지를 변환하세요.", wraplength=280)
        preset_desc.pack(fill=tk.X, padx=10, pady=(0, 10))
        
        # 프리셋 버튼들
        preset_frame = ttk.Frame(controls_frame)
        preset_frame.pack(fill=tk.X, padx=10, pady=10)
        
        ttk.Button(preset_frame, text="기본 만화", command=self.preset_default).pack(fill=tk.X, pady=5)
        ttk.Button(preset_frame, text="강한 엣지", command=self.preset_strong_edges).pack(fill=tk.X, pady=5)
        ttk.Button(preset_frame, text="고품질 만화", command=self.preset_strong_cartoon).pack(fill=tk.X, pady=5)
        
        # 설명 추가
        info_frame = ttk.LabelFrame(controls_frame, text="스타일 설명")
        info_frame.pack(fill=tk.X, padx=10, pady=20)
        
        ttk.Label(info_frame, text="기본 만화: 표준 만화 스타일", wraplength=280).pack(anchor=tk.W, padx=5, pady=2)
        ttk.Label(info_frame, text="강한 엣지: 윤곽선이 두드러진 스타일", wraplength=280).pack(anchor=tk.W, padx=5, pady=2)
        ttk.Label(info_frame, text="고품질 만화: 색상과 윤곽이 강화된 만화", wraplength=280).pack(anchor=tk.W, padx=5, pady=2)
        
        # 하단 상태 바
        bottom_frame = ttk.Frame(main_frame)
        bottom_frame.pack(fill=tk.X, pady=(10, 0))
        
        self.info_label = ttk.Label(bottom_frame, text="NALDA 만화 렌더링 v1.0", foreground=self.colors['text_light'])
        self.info_label.pack(side=tk.LEFT, padx=10)
    
    def open_image(self):
        file_path = filedialog.askopenfilename(
            title="이미지 파일 선택",
            filetypes=[("이미지 파일", "*.jpg *.jpeg *.png *.bmp")]
        )
        
        if file_path:
            try:
                self.filename = file_path
                self.original_image = cv2.imread(file_path)
                if self.original_image is None:
                    self.status_label.configure(text="이미지를 로드할 수 없습니다", foreground=self.colors['danger'])
                    return
                
                self.status_label.configure(text="이미지 로드됨", foreground=self.colors['success'])
                self.info_label.configure(text=f"파일: {os.path.basename(file_path)} | 크기: {self.original_image.shape[1]}x{self.original_image.shape[0]}")
                
                # 이미지 표시
                self.show_image("original")
                
                # 자동 변환
                self.convert_to_cartoon()
            except Exception as e:
                self.status_label.configure(text=f"오류: {str(e)}", foreground=self.colors['danger'])
    
    def convert_to_cartoon(self):
        if self.original_image is None:
            self.status_label.configure(text="먼저 이미지를 로드하세요", foreground=self.colors['warning'])
            return
        
        try:
            self.status_label.configure(text="변환 중...", foreground=self.colors['primary'])
            self.root.update()
            
            # 만화 변환 알고리즘 적용
            self.cartoon_image = self.cartoonize_image(self.original_image)
            
            self.status_label.configure(text="변환 완료", foreground=self.colors['success'])
            
            # 결과 이미지 표시
            self.show_image("cartoon")
        except Exception as e:
            self.status_label.configure(text=f"변환 오류: {str(e)}", foreground=self.colors['danger'])
    
    def cartoonize_image(self, img):
        # 그레이스케일 변환
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 노이즈 제거를 위한 미디언 블러
        gray = cv2.medianBlur(gray, self.median_ksize)
        
        # 엣지 검출 (Canny 또는 적응형 임계값)
        if self.edge_threshold1 < self.edge_threshold2:
            # Canny 엣지 검출
            edges = cv2.Canny(gray, self.edge_threshold1, self.edge_threshold2)
            # 엣지를 두껍게 만들기
            kernel = np.ones((2, 2), np.uint8)
            edges = cv2.dilate(edges, kernel, iterations=1)
            edges = cv2.bitwise_not(edges)  # 반전 (흰색 배경, 검은색 엣지)
        else:
            # 적응형 임계값 사용
            edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
                                         cv2.THRESH_BINARY, self.adaptive_blocksize, self.adaptive_c)
        
        # 색상 평활화 (양방향 필터)
        color = cv2.bilateralFilter(img, self.bilateral_d, self.bilateral_color, self.bilateral_space)
        
        # 색상 양자화 (K-means 클러스터링)
        Z = color.reshape((-1, 3))
        Z = np.float32(Z)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        K = 8  # 색상 수
        _, labels, centers = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        centers = np.uint8(centers)
        res = centers[labels.flatten()]
        quantized = res.reshape((color.shape))
        
        # 엣지와 색상 결합
        edges_3channel = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        cartoon = cv2.bitwise_and(quantized, edges_3channel)
        
        return cartoon
    
    def show_image(self, image_type):
        if image_type == "original" and self.original_image is not None:
            img = self.original_image
            self.current_display = "original"
        elif image_type == "cartoon" and self.cartoon_image is not None:
            img = self.cartoon_image
            self.current_display = "cartoon"
        else:
            return
        
        # OpenCV BGR에서 RGB로 변환
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Tkinter에 표시하기 위해 PIL로 변환
        pil_img = Image.fromarray(img_rgb)
        
        # 화면 크기에 맞게 리사이즈
        display_width = 800
        display_height = int(display_width * (img.shape[0] / img.shape[1]))
        pil_img = pil_img.resize((display_width, display_height), Image.LANCZOS)
        
        # PIL에서 ImageTk로 변환
        img_tk = ImageTk.PhotoImage(image=pil_img)
        
        # 라벨에 이미지 업데이트
        self.image_label.configure(image=img_tk)
        self.image_label.image = img_tk  # 참조 유지
    
    def save_image(self):
        if self.cartoon_image is None:
            self.status_label.configure(text="저장할 이미지가 없습니다", foreground=self.colors['warning'])
            return
        
        if self.filename:
            base_name = os.path.basename(self.filename)
            name, ext = os.path.splitext(base_name)
            save_path = filedialog.asksaveasfilename(
                title="이미지 저장",
                initialdir=self.output_dir,
                initialfile=f"{name}_cartoon{ext}",
                defaultextension=ext,
                filetypes=[("JPEG 파일", "*.jpg"), ("PNG 파일", "*.png"), ("모든 파일", "*.*")]
            )
            
            if save_path:
                try:
                    cv2.imwrite(save_path, self.cartoon_image)
                    self.status_label.configure(text=f"이미지 저장됨: {os.path.basename(save_path)}", foreground=self.colors['success'])
                except Exception as e:
                    self.status_label.configure(text=f"저장 오류: {str(e)}", foreground=self.colors['danger'])
    
    # 프리셋 함수들
    def preset_default(self):
        self.edge_threshold1 = 100
        self.edge_threshold2 = 200
        self.bilateral_d = 9
        self.bilateral_color = 250
        self.bilateral_space = 250
        self.median_ksize = 5
        self.adaptive_blocksize = 9
        self.adaptive_c = 9
        
        self.status_label.configure(text="기본 프리셋 적용됨", foreground=self.colors['primary'])
    
    def preset_strong_edges(self):
        self.edge_threshold1 = 50
        self.edge_threshold2 = 150
        self.bilateral_d = 7
        self.bilateral_color = 150
        self.bilateral_space = 150
        self.median_ksize = 5
        self.adaptive_blocksize = 7
        self.adaptive_c = 5
        
        self.status_label.configure(text="강한 엣지 프리셋 적용됨", foreground=self.colors['primary'])
    
    def preset_strong_cartoon(self):
        self.edge_threshold1 = 70
        self.edge_threshold2 = 200
        self.bilateral_d = 7
        self.bilateral_color = 150
        self.bilateral_space = 150
        self.median_ksize = 5
        self.adaptive_blocksize = 9
        self.adaptive_c = 2
        
        self.status_label.configure(text="강한 만화 프리셋 적용됨", foreground=self.colors['primary'])
    
    def on_closing(self):
        self.root.destroy()

def main():
    try:
        root = tk.Tk()
        app = NALDACartoonRenderer(root)
        root.mainloop()
    except Exception as e:
        import traceback
        print(f"프로그램 실행 중 오류가 발생했습니다: {e}")
        print(traceback.format_exc())

if __name__ == "__main__":
    main() 