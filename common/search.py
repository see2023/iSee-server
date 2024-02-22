import aiohttp
import asyncio
import os
import sys
import logging
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dotenv import load_dotenv
load_dotenv()
from bs4 import BeautifulSoup

SUBSCRIPTION_KEY = os.getenv('AZURE_SUBSCRIPTION_KEY')


async def fetch_page_async(url, session):
    page_res = await session.get(url)
    logging.debug(url, "Page response status: " + str(page_res.status))
    text = await page_res.text()
    # remove script and style tags
    text = text.replace("<script", "<!--").replace("</script>", "-->")
    text = text.replace("<style", "<!--").replace("</style>", "-->")
    # extract body text
    start = text.find("<body")
    end = text.find("</body>")
    content = text[start:end]
    # remove html tags
    content = content.replace("<", " <").replace(">", "> ")
    content = " ".join(content.split())
    content = content.replace("<", "").replace(">", "")
    # remove extra spaces and newlines
    content = content.replace("\n", " ").replace("\r", " ")
    content = " ".join(content.split())
    return content

async def bing_search_by_api(query, count=2, get_page_count=0, offset=0, just_snippet=True):
    output = ""
    try:
        async with aiohttp.ClientSession() as session:
            response = await session.get(
                "https://api.bing.microsoft.com/v7.0/search?q=" + query + "&count=" + str(count) + "&offset=" + str(offset),
                headers={"Ocp-Apim-Subscription-Key": SUBSCRIPTION_KEY}
            )
            body = await response.json()
            # copy name, url, snippet, language, dateLastCrawled from body.webPages to output
            output = [item for item in body["webPages"]["value"]]
            # fetch page contents
            for i in range(get_page_count):
                if i < len(body["webPages"]["value"]):
                    logging.debug("fetching page: " + str(i), body["webPages"]["value"][i]["url"])
                    content= await fetch_page_async(body["webPages"]["value"][i]["url"], session)
                    output[i]["content"] = content
        logging.debug(output)
        if just_snippet:
            # join all snippet to one string
            output = " ".join([item["snippet"] for item in output])
        return output
    except Exception as error:
        logging.error("bing api Error: " + str(error))
        return output


ABSTRACT_MAX_LENGTH = 300 
async def baidu_search(query, count=3, just_abstract=True):
    # 百度上都没搜索到他家的API... 参考第三方：
    # https://github.com/amazingcoderxyz/python-baidusearch/blob/master/baidusearch/baidusearch.py parse_html
#   -H 'Referer: https://www.baidu.com/' \
#   -H 'User-Agent: Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36' \
    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36',
        'Referer': 'https://www.baidu.com/'
    }
    query = query.replace(" ", "+")

    url = "https://www.baidu.com/s?ie=utf-8&tn=baidu&wd=" + query
    parsed_count = 0

    output = ""
    try:
        async with aiohttp.ClientSession(headers=headers) as session:
            response = await session.get(url)
            text = await response.text()
            root = BeautifulSoup(text, "lxml")
            div_contents = root.find("div", id="content_left")
            for div in div_contents.contents:
                if type(div) != type(div_contents):
                    continue

                class_list = div.get("class", [])
                if not class_list:
                    continue

                if "c-container" not in class_list:
                    continue

                title = ''
                url = ''
                abstract = ''
                try:
                    # 遍历所有找到的结果，取得标题和概要内容（50字以内）
                    if "xpath-log" in class_list:
                        if div.h3:
                            title = div.h3.text.strip()
                            url = div.h3.a['href'].strip()
                        else:
                            title = div.text.strip().split("\n", 1)[0]
                            if div.a:
                                url = div.a['href'].strip()

                        if div.find("div", class_="c-abstract"):
                            abstract = div.find("div", class_="c-abstract").text.strip()
                        elif div.div:
                            abstract = div.div.text.strip()
                        else:
                            abstract = div.text.strip().split("\n", 1)[1].strip()
                    elif "result-op" in class_list:
                        if div.h3:
                            title = div.h3.text.strip()
                            url = div.h3.a['href'].strip()
                        else:
                            title = div.text.strip().split("\n", 1)[0]
                            url = div.a['href'].strip()
                        if div.find("div", class_="c-abstract"):
                            abstract = div.find("div", class_="c-abstract").text.strip()
                        elif div.div:
                            abstract = div.div.text.strip()
                        else:
                            # abstract = div.text.strip()
                            abstract = div.text.strip().split("\n", 1)[1].strip()
                    else:
                        if div.get("tpl", "") != "se_com_default":
                            if div.get("tpl", "") == "se_st_com_abstract":
                                if len(div.contents) >= 1:
                                    title = div.h3.text.strip()
                                    if div.find("div", class_="c-abstract"):
                                        abstract = div.find("div", class_="c-abstract").text.strip()
                                    elif div.div:
                                        abstract = div.div.text.strip()
                                    else:
                                        abstract = div.text.strip()
                            else:
                                if len(div.contents) >= 2:
                                    if div.h3:
                                        title = div.h3.text.strip()
                                        url = div.h3.a['href'].strip()
                                    else:
                                        title = div.contents[0].text.strip()
                                        url = div.h3.a['href'].strip()
                                    # abstract = div.contents[-1].text
                                    if div.find("div", class_="c-abstract"):
                                        abstract = div.find("div", class_="c-abstract").text.strip()
                                    elif div.div:
                                        abstract = div.div.text.strip()
                                    else:
                                        abstract = div.text.strip()
                        else:
                            if div.h3:
                                title = div.h3.text.strip()
                                url = div.h3.a['href'].strip()
                            else:
                                title = div.contents[0].text.strip()
                                url = div.h3.a['href'].strip()
                            if div.find("div", class_="c-abstract"):
                                abstract = div.find("div", class_="c-abstract").text.strip()
                            elif div.div:
                                abstract = div.div.text.strip()
                            else:
                                abstract = div.text.strip()
                except Exception as e:
                    logging.debug("catch exception duration parsing page html, e={}".format(e))
                    continue

                if ABSTRACT_MAX_LENGTH and len(abstract) > ABSTRACT_MAX_LENGTH:
                    abstract = abstract[:ABSTRACT_MAX_LENGTH]
                # 去除多余的空格和换行符
                title = " ".join(title.split())
                abstract = " ".join(abstract.split())

                if just_abstract:
                    item = abstract + "\n"
                else:
                    item = "title:{}, abstract:{}\n".format(title, abstract)
                logging.debug("count:{}, item:{}".format(parsed_count, item))
                output += item
                parsed_count += 1
                if parsed_count >= count:
                    break
            return output
    except Exception as error:
        logging.error("baidu search Error: " + str(error))
        return output

async def search(query, count=3):
    # 先搜索百度，如果失败，再调用付费bing api
    output = await baidu_search(query, count)
    if not output or len(output) < 10:
        output = await bing_search_by_api(query, count, 0, 0, True)
    return output

async def test():
    # output = await bing_search_by_api("gpt-sovits", 1, 0, 0)
    # logging.info(output)
    # output = await baidu_search("从百草园到三味书屋")
    output = await search("松江天气")
    logging.info(output)

if __name__ == '__main__':
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(levelname)s [in %(pathname)s:%(lineno)d] - %(message)s',
    )
    asyncio.run(test())
