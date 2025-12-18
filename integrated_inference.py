import os
import torch
import argparse
import yaml
import numpy as np
from PIL import Image
from tqdm import tqdm
from transformers import BertTokenizer
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, DDPMScheduler

# 导入项目内部组件
from models.poem2layout import Poem2LayoutGenerator
from inference.greedy_decode import greedy_decode_poem_layout
from stage2_generation.utils.ink_mask import InkWashMaskGenerator
from data.visualize import draw_layout

# 定义完整的 50 句测试诗集
POEMS_50 = [
    "白日依山尽，黄河入海流。", "明月松间照，清泉石上流。", "野旷天低树，江清月近人。",
    "两岸青山相对出，孤帆一片日边来。", "孤舟蓑笠翁，独钓寒江雪。", "大漠孤烟直，长河落日圆。",
    "山高月小，水落石出。", "月落乌啼霜满天，江枫渔火对愁眠。", "落霞与孤鹜齐飞，秋水共长天一色。",
    "渭城朝雨浥轻尘，客舍青青柳色新。", "千山鸟飞绝，万径人踪灭。", "小楼一夜听春雨，深巷明朝卖杏花。",
    "竹喧归浣女，莲动下渔舟。", "云想衣裳花想容，春风拂槛露华浓。", "独在异乡为异客，每逢佳节倍思亲。",
    "江流天地外，山色有无中。", "青山横北郭，白水绕东城。", "柴门闻犬吠，风雪夜归人。",
    "空山新雨后，天气晚来秋。", "一水护田将绿绕，两山排闼送青来。", "接天莲叶无穷碧，映日荷花别样红。",
    "黄河远上白云间，一片孤城万仞山。", "山回路转不见君，雪上空留马行处。", "西塞山前白鹭飞，桃花流水鳜鱼肥。",
    "日出江花红胜火，春来江水绿如蓝。", "两岸猿声啼不住，轻舟已过万重山。", "溪云初起日沉阁，山雨欲来风满楼。",
    "鸡声茅店月，人迹板桥霜。", "林表明霁色，城中增暮寒。", "清明时节雨纷纷，路上行人欲断魂。",
    "轻舟短棹西湖好，绿水逶迤，芳草长堤。", "山光悦鸟性，潭影空人心。", "绿树村边合，青山郭外斜。",
    "霜落熊升树，林空鹿饮溪。", "千峰笋石千株玉，万树松萝万朵云。", "烟波江上使人愁。",
    "渔舟逐水爱山春，两岸桃花夹古津。", "楼观沧海日，门对浙江潮。", "松风吹解带，山月照弹琴。",
    "野渡无人舟自横。", "湖光秋月两相和，潭面无风镜未磨。", "江碧鸟逾白，山青花欲燃。",
    "石泉流暗壁，草露滴秋根。", "晓看红湿处，花重锦官城。", "榆柳荫后檐，桃李罗堂前。",
    "木末芙蓉花，山中发红萼。", "露从今夜白，月是故乡明。", "萧萧梧叶送寒声，江上秋风动客情。",
    "山寺月中寻桂子，郡亭枕上看潮头。", "横看成岭侧成峰，远近高低各不同。"
]

def parse_args():
    parser = argparse.ArgumentParser(description="Poem2Ink: 50句诗歌全自动批量推理（彩色语义强绑定版）")
    parser.add_argument("--output_base", type=str, default="./inference_results_v50_color", help="结果保存的根目录")
    parser.add_argument("--layout_ckpt", type=str, required=True, help="强化学习后的布局模型路径")
    parser.add_argument("--taiyi_model_path", type=str, required=True, help="本地太乙模型路径")
    parser.add_argument("--lora_path", type=str, required=True, help="微调后的 LoRA 权重目录")
    parser.add_argument("--controlnet_seg_path", type=str, required=True, help="语义分割控制网路径 (对应彩色Mask)")
    parser.add_argument("--controlnet_t_path", type=str, required=True, help="ControlNet 风格流路径")
    return parser.parse_args()

def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # --- 1. 初始化模型 ---
    print("\n[Init] 正在加载布局模型 (Stage 1)...")
    with open("configs/default.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    layout_model = Poem2LayoutGenerator(
        bert_path=config['model']['bert_path'],
        num_classes=config['model']['num_classes'],
        hidden_size=config['model']['hidden_size'],
        bb_size=config['model']['bb_size'],
        decoder_layers=config['model']['decoder_layers'],
        decoder_heads=config['model']['decoder_heads'],
        latent_dim=config['model'].get('latent_dim', 64)
    )
    
    ckpt = torch.load(args.layout_ckpt, map_location=device)
    layout_model.load_state_dict(ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt)
    layout_model.to(device).eval()
    tokenizer_bert = BertTokenizer.from_pretrained(config['model']['bert_path'])

    print("[Init] 正在加载太乙生成模型与语义控制网 (Stage 2)...")
    # 核心修改：这里加载的是支持彩色语义分割输入的 ControlNet
    controlnet_seg = ControlNetModel.from_pretrained(args.controlnet_seg_path, torch_dtype=torch.float16)
    controlnet_t = ControlNetModel.from_pretrained(args.controlnet_t_path, torch_dtype=torch.float16)
    
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        args.taiyi_model_path,
        controlnet=[controlnet_seg, controlnet_t],
        torch_dtype=torch.float16,
        local_files_only=True 
    ).to(device)
    
    pipe.load_lora_weights(args.lora_path)
    pipe.scheduler = DDPMScheduler.from_config(pipe.scheduler.config)
    
    # 初始化彩色 Mask 生成器
    ink_gen = InkWashMaskGenerator(width=512, height=512)

    # --- 2. 批量推理循环 ---
    print(f"\n🚀 开始彩色语义强绑定推理 50 句诗歌...")
    
    for i, poem in enumerate(tqdm(POEMS_50, desc="Overall Progress")):
        poem_clean = poem[:12].replace("，", "_").replace("。", "").replace("？", "").replace("！", "").strip()
        save_dir = os.path.join(args.output_base, f"{i+1:02d}_{poem_clean}")
        os.makedirs(save_dir, exist_ok=True)

        try:
            # STEP 1: 布局生成
            layout = greedy_decode_poem_layout(
                layout_model, tokenizer_bert, poem, 
                max_elements=30, device=device, mode="sample"
            )
            
            # 保存热力图
            heatmap_temp_path = f"outputs/heatmaps/integrated_{poem_clean}流.png"
            if os.path.exists(heatmap_temp_path):
                os.rename(heatmap_temp_path, os.path.join(save_dir, "01_heatmap.png"))

            # 保存布局草图 (确保这里的颜色与 ink_mask 一致)
            draw_layout(layout, f"RL Inference: {poem}", os.path.join(save_dir, "01_layout.png"))

            # STEP 2: 彩色语义 Mask 转换 [核心创新点修改]
            # 现在生成的 mask_img 是彩色的 RGB 图像
            mask_img = ink_gen.convert_boxes_to_mask(layout)
            mask_img.save(os.path.join(save_dir, "02_semantic_color_mask.png"))

            # STEP 3: 语义强绑定 Prompt 构建 [骚操作]
            # 为了强化颜色与意象的绑定，我们在提示词中加入颜色引导描述
            semantic_binding_hints = (
                "，红色区域画山，蓝色区域画水，绿色区域画树，黄色区域画建筑，"
                "紫色区域画花卉，青色区域画人物，橙色区域画飞鸟"
            )
            style_suffix = "，写意水墨画，中国画风格，杰作，高分辨率，层次分明"
            
            full_prompt = f"{poem}{semantic_binding_hints}{style_suffix}"
            
            # 最终山水画生成
            # 传入彩色 Mask 作为控制信号
            final_image = pipe(
                prompt=full_prompt,
                image=[mask_img, mask_img], # 两路 ControlNet 均以此彩色语义为基准
                num_inference_steps=35,
                guidance_scale=8.5,
                controlnet_conditioning_scale=[1.2, 0.7] # 调高语义流权重以增强“强绑定”
            ).images[0]
            
            final_image.save(os.path.join(save_dir, "03_final_painting.png"))

        except Exception as e:
            print(f"\n❌ [Error] 诗句 '{poem}' 处理失败: {e}")
            continue

    print(f"\n✨ 任务圆满完成！全部结果已保存在: {args.output_base}")

if __name__ == "__main__":
    main()