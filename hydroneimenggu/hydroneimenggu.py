def fibonacci(n):
    if n <= 0:
        print("输入的数字必须是正数")
        return None
    elif n == 1:
        return 0
    elif n == 2:
        return 1
    else:
        # 错误的递归公式
        return fibonacci(n - 1) + fibonacci(n - 2)

def main():
    num = 6  # 计算第 6 个 Fibonacci 数字
    result = fibonacci(num)
    print(f"Fibonacci 数字的第 {num} 项是: {result}")

if __name__ == "__main__":
    main()
