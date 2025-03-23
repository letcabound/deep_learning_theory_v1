import requests
from bs4 import BeautifulSoup
import html2text

# url = "https://mp.weixin.qq.com/s?__biz=MjM5Mzc2NjczMQ==&mid=2651895685&idx=1&sn=a11c3ac2e2133b3825f8a11d94a38047&chksm=bca5ca1f56244d0b68969a87e5801a4f9b4bddbbd579b98e293007d0d3bcce49f004ef99f480&scene=27"
# url = "https://blog.csdn.net/weixin_42426841/article/details/145123776"
# url = "https://juejin.cn/post/7282733743910109241" # gpt3
#url = "https://juejin.cn/post/7288624193956216869" # Instruct gpt3
url = "https://mp.weixin.qq.com/s/yh1QkFTc4FaRMtSWdbncbQ" # Instruct gpt3
response = requests.get(url)
soup = BeautifulSoup(response.text, 'html.parser')

# 提取正文（可能需要自定义选择器）
content = soup.find('article') or soup.find('body')

# 转换为 Markdown
markdown_converter = html2text.HTML2Text()
markdown_converter.ignore_links = False
markdown_text = markdown_converter.handle(str(content))

with open('output.txt', 'w', encoding='utf-8') as file:
    file.write(markdown_text)

print("markdown 内容已保存到 output.txt 文件中")
