from pdf2image import convert_from_path
from PIL import Image

# 选择三个PDF文件
pdf_files = [
    # "/home/llm/FineGrainedRLHF/Pictures/2D-a-v2/fact-reward_raw.pdf",
    # "/home/llm/FineGrainedRLHF/Pictures/2D-a-v2/rel-reward_raw.pdf",
    # "/home/llm/FineGrainedRLHF/Pictures/2D-a-v2/comp-reward_raw.pdf",
    "/home/llm/FineGrainedRLHF/Pictures/2D-a-v2/fact-reward_KL.pdf",
    "/home/llm/FineGrainedRLHF/Pictures/2D-a-v2/rel-reward_KL.pdf",
    "/home/llm/FineGrainedRLHF/Pictures/2D-a-v2/comp-reward_KL.pdf",
]

# 设置统一的尺寸
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

# 生成从左往右排列的图片
horizontal_grid_width = 3 * standard_width
horizontal_grid_height = standard_height

horizontal_grid_image = Image.new('RGB', (horizontal_grid_width, horizontal_grid_height), (255, 255, 255))

# 将每张图片粘贴到横排图片中
for i, image in enumerate(images):
    x = i * standard_width
    y = 0
    horizontal_grid_image.paste(image, (x, y))

# horizontal_grid_image.save('/home/llm/FineGrainedRLHF/Pictures/2D-NSB/RR-SSB_image_horizontal.pdf', 'PDF', quality=100)
horizontal_grid_image.save('/home/llm/FineGrainedRLHF/Pictures/2D-NSB/KL-SSB_image_horizontal.pdf', 'PDF', quality=100)

# 生成从上往下排列的图片
vertical_grid_width = standard_width
vertical_grid_height = 3 * standard_height

vertical_grid_image = Image.new('RGB', (vertical_grid_width, vertical_grid_height), (255, 255, 255))

# 将每张图片粘贴到竖排图片中
for i, image in enumerate(images):
    x = 0
    y = i * standard_height
    vertical_grid_image.paste(image, (x, y))

# vertical_grid_image.save('/home/llm/FineGrainedRLHF/Pictures/2D-NSB/RR-SSB_image_vertical.pdf', 'PDF', quality=100)
vertical_grid_image.save('/home/llm/FineGrainedRLHF/Pictures/2D-NSB/KL-SSB_image_vertical.pdf', 'PDF', quality=100)


# 展示生成的图片（如果你想在本地查看）
horizontal_grid_image.show()
vertical_grid_image.show()
