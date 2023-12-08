import os
import json
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain.agents import initialize_agent, AgentType
from langchain.chat_models import ChatOpenAI
from langchain.prompts import MessagesPlaceholder
from langchain.memory import ConversationSummaryBufferMemory
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
from langchain.tools import BaseTool
from langchain.schema import SystemMessage
from fastapi import FastAPI
from pydantic import BaseModel

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI()

# Define the request body model
class URLQuery(BaseModel):
    url: str

# Web scraping function
def scrape_website(url: str):
    browserless_api_key = os.getenv("BROWSERLESS_API_KEY")
    headers = {'Cache-Control': 'no-cache', 'Content-Type': 'application/json'}
    data = {"url": url}
    data_json = json.dumps(data)
    post_url = f"https://chrome.browserless.io/content?token={browserless_api_key}"
    response = requests.post(post_url, headers=headers, data=data_json)

    if response.status_code == 200:
        soup = BeautifulSoup(response.content, "html.parser")
        for script in soup(["script", "style"]):
            script.decompose()
        text = soup.get_text()
        return text
    else:
        print(f"HTTP request failed with status code {response.status_code}")
        return None
    
# Summary function using Langchain
def summary(content):
    llm = ChatOpenAI(temperature=0, model="gpt-4-1106-preview")
    text_splitter = RecursiveCharacterTextSplitter(separators=["\n\n", "\n"], chunk_size=10000, chunk_overlap=500)
    docs = text_splitter.create_documents([content])

    map_prompt = """
    Write a summary of the following text for {content} using about 60 words. The text is Scraped data from a website so 
    will have a lot of useless information that doesn't relate to this topic, links, other news stories etc.. 
    Only summarise the relevant Info and try to keep as much factual information Intact.
    "{text}"
    SUMMARY:
    """
    map_prompt_template = PromptTemplate(template=map_prompt, input_variables=["text", "content"])

    summary_chain = load_summarize_chain(llm=llm, chain_type='map_reduce', map_prompt=map_prompt_template, combine_prompt=map_prompt_template, verbose=True)

    output = summary_chain.run(input_documents=docs, content=content)
    return output

@app.post("/summarize-url")
def summarize_url(query: URLQuery):
    scraped_content = scrape_website(query.url)
    if scraped_content:
        summarized_content = summary(scraped_content)
        # Append the URL to the summarized content
        final_output = summarized_content + "\n\n"+ query.url
        return {"summarized_content": final_output}
    else:
        return {"error": "Failed to scrape the website or no content to summarize"}

# Uvicorn main to run the app
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

# run locally $ uvicorn main:app --reload
