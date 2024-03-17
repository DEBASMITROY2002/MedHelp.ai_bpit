import copy
import datetime
import numpy as np
import os
import openai
import time
import string
import re
import collections
import tiktoken
import json

from datetime import datetime as dt
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from collections import defaultdict
from typing import Optional, Type, List, Tuple
from pydantic import BaseModel, Field
from getpass import getpass
import PIL

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain_openai import OpenAI
from langchain.agents import Tool,initialize_agent, AgentType, ZeroShotAgent, AgentExecutor
from langchain.tools import BaseTool
from langchain.chat_models import ChatOpenAI
from langchain.output_parsers import PydanticOutputParser
from langchain.chains import LLMChain, LLMMathChain, TransformChain, SequentialChain,  create_extraction_chain, create_extraction_chain_pydantic, create_tagging_chain, create_tagging_chain_pydantic
from langchain_community.tools.pubmed.tool import PubmedQueryRun


from semantic_router import Route
from semantic_router.encoders import OpenAIEncoder
from semantic_router.layer import RouteLayer


from langchain.chains.question_answering import load_qa_chain
from langchain.memory import ConversationBufferMemory
from langchain.docstore.document import Document

from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper

from transformers import (
    pipeline,
    AutoTokenizer,
    AutoModelForSequenceClassification,
    CLIPProcessor,
    CLIPModel,
    AutoModelForMaskedLM
    )
import torch


""""
What is the medicine of Dengue ? 

Hi how are you ?

Is it a case of serious blood loss?

Is fast aid is enough ?
"""