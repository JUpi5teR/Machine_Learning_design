import argparse
from train import train_model
from predict import generate_submission


def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="面部情感分类程序")
    parser.add_argument(
        '--mode',
        type=str,
        required=True,
        choices=['train', 'predict'],
        help='运行模式：train（训练模型）或 predict（生成提交文件）'
    )

    args = parser.parse_args()

    # 根据模式执行相应功能
    if args.mode == 'train':
        print("=== 运行模式：训练模型 ===")
        train_model()
    elif args.mode == 'predict':
        print("=== 运行模式：生成提交文件 ===")
        generate_submission()


if __name__ == "__main__":
    main()