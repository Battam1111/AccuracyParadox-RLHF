from PyPDF2 import PdfFileReader
from pdf2image import convert_from_path
from PIL import Image

# 将PDF文件转换为图像
pdf_files = [
    "/home/llm/FineGrainedRLHF/Pictures/2D-a-v2/fact-reward_raw.pdf",
    "/home/llm/FineGrainedRLHF/Pictures/2D-a-v2/fact-reward_KL.pdf",
    "/home/llm/FineGrainedRLHF/Pictures/2D-a-v2/fact-reward_penalized.pdf",
    "/home/llm/FineGrainedRLHF/Pictures/2D-a-v2/rel-reward_raw.pdf",
    "/home/llm/FineGrainedRLHF/Pictures/2D-a-v2/rel-reward_KL.pdf",
    "/home/llm/FineGrainedRLHF/Pictures/2D-a-v2/rel-reward_penalized.pdf",
    "/home/llm/FineGrainedRLHF/Pictures/2D-a-v2/comp-reward_raw.pdf",
    "/home/llm/FineGrainedRLHF/Pictures/2D-a-v2/comp-reward_KL.pdf",
    "/home/llm/FineGrainedRLHF/Pictures/2D-a-v2/comp-reward_penalized.pdf"
]

# 设置统一的尺寸 (假设所有图片的尺寸相同，如果不相同则需要统一调整大小)
standard_width = 2000  # 你可以根据需要调整
standard_height = 1000  # 你可以根据需要调整

images = []

# 将PDF文件转换为图像，并统一调整大小
for pdf_file in pdf_files:
    image = convert_from_path(pdf_file)[0]
    image = image.resize((standard_width, standard_height), Image.Resampling.LANCZOS)
    
    # 处理透明背景，转为白色
    background = Image.new("RGB", (standard_width, standard_height), (255, 255, 255))
    background.paste(image, mask=image.split()[3] if image.mode == 'RGBA' else None)
    
    images.append(background)

# 假设所有图像的尺寸相同
width, height = images[0].size
grid_width = 3 * width
grid_height = 3 * height

# 创建一个新的白色背景的空白图像
grid_image = Image.new('RGB', (grid_width, grid_height), (255, 255, 255))

# 将每张图片粘贴到九宫格的正确位置
for i, image in enumerate(images):
    x = (i % 3) * width
    y = (i // 3) * height
    grid_image.paste(image, (x, y))

# 将拼接后的图像保存为PDF格式
grid_image.save('/home/llm/FineGrainedRLHF/Pictures/2D-NSB/NSB_image.pdf', 'PDF', quality=100)
# grid_image.save('/home/llm/FineGrainedRLHF/Pictures/2D-NSB/NSB_image-small.pdf', 'PDF', quality=100)
# grid_image.save('/home/llm/FineGrainedRLHF/Pictures/2D-NSB/NSB_image-base.pdf', 'PDF', quality=100)
# grid_image.save('/home/llm/FineGrainedRLHF/Pictures/2D-NSB/NSB_image-large.pdf', 'PDF', quality=100)

grid_image.show()
