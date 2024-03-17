from common_utils import *


class SparseModel:
  def __init__(self):
    self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    self.model = AutoModelForMaskedLM.from_pretrained("naver/splade-cocondenser-selfdistil").to(self.device)
    self.sparse_tokenizer = AutoTokenizer.from_pretrained("naver/splade-cocondenser-selfdistil",padding=True,truncation=True)

  def decode_sparse_dict(self, sparse_dict,trim=None):
    a = np.zeros((30522))
    a[sparse_dict['indices']] = sparse_dict['values']
    if trim is not None:
      a[a.argsort()[:-trim]] = 0
    return a

  def decode_sparse_dicts(self, sparse_dicts,trim=None):
    res = []
    for _ in sparse_dicts:
      res.append(self.decode_sparse_dict(_,trim).tolist())
    return res

  def formalize(self, sparse_dict):
    idx2token = {idx: token for token, idx in self.sparse_tokenizer.get_vocab().items()}
    sparse_dict_tokens = {
        idx2token[idx]: weight for idx, weight in zip(sparse_dict['indices'], sparse_dict['values'])
    }
    sparse_dict_tokens = {
        k: v for k, v in sorted(
            sparse_dict_tokens.items(),
            key=lambda item: item[1],
            reverse=True
        )
    }
    return sparse_dict_tokens

  def __call__(self, texts:list):
    sparse_dicts = []
    sparse_vecs = []

    for text in tqdm(texts):
        input_ids = self.sparse_tokenizer(text, return_tensors='pt').to(self.device)

        with torch.no_grad():
          logits = self.model(**input_ids).logits

        sparse_vec = torch.max(
            torch.log(
                1+torch.relu(logits)
            )*input_ids.attention_mask.unsqueeze(-1),
        dim=1)[0].squeeze();

        # sparse_vec = sparse_vec.cpu().numpy()

        indices = sparse_vec.nonzero().squeeze().cpu().tolist()
        values = sparse_vec[indices].cpu().tolist()
        sparse_dict = {'indices': indices, 'values': values}
        sparse_dicts.append(sparse_dict)
        sparse_vecs.append(np.array(sparse_vec.cpu()).tolist())

    return sparse_dicts


def run_symptom_tagger(text):
  # keywords = review['keywords']

  symp_schema = {
      "properties": {
          "is_symptom": {
              "type": "string",
              "description": "Indicates whether the text contains any user provided symptoms. 'Yes' means the text contains symptoms, 'No' means it doesn't.",
              "enum": ['Yes', 'No']
          },
          "symptoms": {
              "type": "array",
              "description": "Array of strings representing symptoms mentioned in the text.",
              "items": {
                  "type": "string"
              }
          }
      },
      "required": ["is_symptom", "symptoms"]
  }


  symp_llm = ChatOpenAI(openai_api_key=os.getenv("OPENAI_API"),temperature=0, model="gpt-3.5-turbo-0613")
  symp_chain = create_tagging_chain(symp_schema, symp_llm)
  symp_output = symp_chain(text)['text']

  return symp_output




def run_filter_tagger(text):
    # List of commonly used medicines and antibiotics
    medicines_and_antibiotics = [
        "Acetaminophen (Paracetamol)",
        "Ibuprofen",
        "Naproxen",
        "Aspirin (Salicylic Acid)",
        "Prednisone",
        "Hydrocodone (Vicodin)",
        "OxyContin (Oxycodone)",
        "Morphine",
        "Fentanyl",
        "Warfarin (Coumadin)",
        "Amoxicillin",
        "Azithromycin (Zithromax)",
        "Ciprofloxacin (Cipro)",
        "Doxycycline",
        "Penicillin",
        "Clindamycin",
        "Erythromycin",
        "Levofloxacin (Levaquin)",
        "Metronidazole (Flagyl)",
        "Trimethoprim-sulfamethoxazole (Bactrim)",
    ]

    filter_schema = {
        "properties": {
            "is_relevant": {
                "type": "string",
                "description": "Indicates whether the conversation is related to medical questions/answers or any queries concerning diseases/medicines. "
                               "'Yes' implies relevance to medical topics. 'No' indicates the absence of medical relevance.",
                "enum": ['Yes', 'No']
            }
        },
        "required": ["is_relevant"]
    }

    filter_llm = ChatOpenAI(openai_api_key=os.getenv("OPENAI_API"), temperature=0.0, model="gpt-3.5-turbo-0613")
    filter_chain = create_tagging_chain(filter_schema, filter_llm)
    filter_output = filter_chain(text)['text']

    return filter_output



def create_routes_for_cluster_to_multiqry(cluster_to_multiqry):
  # Routes for different issues
  routes = []
  for c,qs in cluster_to_multiqry.items():
    route = Route(
        name=c,
        utterances=qs,
    )
    routes.append(route)
  return routes

def sem_route_runner(cluster_to_multiqry):
  routes = create_routes_for_cluster_to_multiqry(cluster_to_multiqry)
  encoder = OpenAIEncoder()
  rl = RouteLayer(encoder=encoder, routes=routes)
  return rl



def get_med_routes():
  rl =sem_route_runner({
      "medicine_route":[
          "I have a little fever. What medicine I can use?",
          "I have a headache for the last 2 days continuously. Can you suggest some medication?",
          "My throat is sore and I have difficulty swallowing. What over-the-counter medicine would you recommend?",
          "I accidentally cut my finger while cooking. What should I use to disinfect it?",
          "I have been feeling nauseous lately. Any suggestions for over-the-counter medicine to relieve nausea?",
          "I twisted my ankle while jogging. What pain reliever would you recommend?",
          "I have a persistent cough with phlegm. What cough syrup would be effective?",
          "My allergies are acting up. Can you suggest an antihistamine?",
          "I have a skin rash that's itchy. What cream or ointment should I apply?",
          "I suffer from occasional migraines. Can you recommend a medication that provides quick relief?",
          "I have trouble sleeping at night. Are there any over-the-counter sleep aids you would recommend?",
          "I accidentally burned my hand. What should I apply to soothe the burn?",
          "I'm experiencing menstrual cramps. What pain reliever is best for relieving them?",
          "I have been feeling constipated lately. Any over-the-counter remedies?",
          "My child has a fever. What dosage of children's fever reducer should I give?",
      ],
      "symptoms_route":[
          "I have a persistent cough with blood in my sputum. What could be the cause?",
          "I feel a sharp pain in my chest when I breathe deeply. What might be causing this?",
          "I have a rash on my skin that's spreading rapidly. What condition could this indicate?",
          "I've been feeling extremely fatigued lately, even after a full night's sleep. What could be the reason?",
          "I'm experiencing sudden weight loss without changing my diet or exercise routine. What might be causing this?",
          "I have a tingling sensation and numbness in my fingers and toes. What could be the underlying issue?",
          "I've been experiencing frequent headaches accompanied by blurred vision. What might be causing this?",
          "I feel a lump in my breast. Should I be concerned about breast cancer?",
          "I'm having difficulty concentrating and remembering things lately. What might be causing this cognitive decline?",
          "I'm experiencing intense abdominal pain, especially after meals. What could be causing this digestive issue?",
          "I have a persistent sore throat and difficulty swallowing. What condition could be causing these symptoms?",
          "I've been experiencing frequent urination and increased thirst lately. What might be causing these symptoms?",
          "I feel dizzy and lightheaded when I stand up quickly. What could be causing this sudden drop in blood pressure?",
          "I'm experiencing joint pain and stiffness, especially in the mornings. What could be causing this?",
          "I have a fever and a rash that looks like small red spots. What could be causing this combination of symptoms?",
      ],
      "general_route":[
          "What are the symptoms and treatment options for dengue fever?",
          "Can you explain the difference between viral and bacterial infections?",
          "What are the risk factors for heart disease?",
          "How does diabetes affect the body?",
          "What are the recommended vaccinations for children?",
          "Can you explain the importance of maintaining a healthy diet and lifestyle?",
          "What are the early signs of Alzheimer's disease?",
          "What is the difference between a cold and the flu?",
          "How does stress impact overall health?",
          "What are the potential complications of untreated hypertension?",
          "Can you explain the benefits of regular exercise?",
          "What is the recommended daily water intake for adults?",
          "What are the symptoms and treatment options for asthma?",
          "Can you explain the concept of herd immunity?",
          "What are the common causes of food poisoning?",
      ]
  })

  return rl


def get_vis_routes():
  rl =sem_route_runner({
      "blood":[
        "I cut my finger while chopping vegetables, and blood gushed out.",
        "After the accident, blood stained his clothes and dripped onto the floor.",
        "She noticed blood in her urine and immediately called her doctor.",
        "The dog's paw was injured, and blood trailed behind him as he limped.",
        "In the crime scene, blood splatters marked the walls and floor.",
        "The wound wasn't deep, but blood oozed out, staining the bandage.",
        "He felt weak and dizzy as blood loss from the wound increased.",
        "The surgeon worked quickly to stop the bleeding and clean the blood.",
        "A small cut on her lip caused blood to trickle down her chin.",
        "The victim's blood pooled beneath his body at the crime scene."
      ]
  })

  return rl



class ContSearchSymp:
  def __init__(self,file_path='./dis_and_symptom.json'):
    with open(file_path, "r") as json_file:
      self.kb = json.load(json_file)
    self.model = SparseModel()

  def search(self, text, top_k = 5):
    text_emb_ = self.model([text])[0]
    text_emb = self.model.decode_sparse_dict(text_emb_)
    sorted_res = sorted(self.kb, key=lambda x: np.dot(text_emb,self.model.decode_sparse_dict(x[2])))[-top_k:]
    sorted_res_no_emb = [[_[0],_[1]] for _ in sorted_res]
    return sorted_res_no_emb

class WikiInputs(BaseModel):
    """Inputs to the wikipedia tool."""

    query: str = Field(
        description="query to look up in Wikipedia, should be 4 or less words . Queries are related medical question answering."
    )

def get_med_wiki_tool():
  api_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=1000)

  tool = WikipediaQueryRun(api_wrapper=api_wrapper)

  tool = WikipediaQueryRun(
      name="wiki-tool",
      description="look up things realted to medical question answering in wikipedia",
      args_schema=WikiInputs,
      api_wrapper=api_wrapper,
      return_direct=True,
  )

  return tool


def get_pumed_tool():
  tool = PubmedQueryRun()
  return tool


def get_llm_tool():
  prompt = PromptTemplate(
    input_variables=["query"],
    template="{query}"
  )

  llm_chain = LLMChain(llm=OpenAI(temperature=0.1), prompt=prompt)

  # initialize the LLM tool
  llm_tool = Tool(
      name='Language Model',
      func=llm_chain.run,
      description='use this tool for general purpose queries and logic'
  )
  return llm_tool


class ClipModelBlood:
  def __init__(self):
    self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    self.texts=[
      "Abrasion, Reddish-pink, minor bleeding",
      "Laceration, Bright red, significant bleeding",
      "Incision, Varies, depending on bleeding",
      "Puncture, Small red spot, minimal bleeding",
      "Avulsion, Deep red, heavy bleeding",
      "Gunshot Wounds, Varies widely, deep red",
      "Contusion, Purplish-red, blue-black, green, yellow",
      "Hematoma, Reddish-purple, swollen area",
      "Arterial Bleeding, Bright red, life-threatening",
      "Venous Bleeding, Darker red, slower flow"
  ]

  def __call__(self,image_path):
    image = PIL.Image.open(image_path)
    inputs = self.processor(self.texts, images=image, return_tensors="pt", padding=True)
    outputs = self.model(**inputs)
    logits_per_image = outputs.logits_per_image
    probs = logits_per_image.softmax(dim=1)
    probs=probs.detach().numpy()
    max_ind=np.argmax(probs)
    return self.texts[max_ind]



class ConvMain:
  def __init__(self):
    template = """You are a chatbot having a conversation with a human regarding medical queries.

    Given the following extracted parts of a long document and a question, create a final answer.

    {context}

    {chat_history}
    Human: {human_input}
    Chatbot:"""

    self.prompt = PromptTemplate(
        input_variables=["chat_history", "human_input", "context"], template=template
    )

    self.reset_chain()

    self.pub_med_toll = get_pumed_tool()
    self.med_wiki_tool = get_med_wiki_tool()

    self.med_routes = get_med_routes()
    self.vis_routes = get_vis_routes()

    self.clip_model_blood = ClipModelBlood()

    self.reset_chain()

  def reset_chain(self):
    self.memory = ConversationBufferMemory(memory_key="chat_history", input_key="human_input")
    self.chain = load_qa_chain(
            OpenAI(temperature=0), chain_type="stuff", memory=self.memory, prompt=self.prompt
        )

  def __call__(self, query, symp_image_path = None):
    route = self.med_routes(query).name
    # visual_route = self.vis_routes(query).name

    if symp_image_path is not None:# and visual_route == 'blood':
      print("blood wound route active")
      vis_text = self.clip_model_blood(symp_image_path)
      add_qry = f"\n I have blood wound : <{vis_text}>"
      query+=add_qry

    if route=='medicine_route' or route=='symptoms_route':
      context = self.pub_med_toll.invoke(query)
    elif route=='general_route':
      context = self.med_wiki_tool.invoke(query) + '\n' + self.pub_med_toll.invoke(query)
    else:
      print("Warning: High Chances of not realted to med qry")
      context = ""

    print("-------------- Context --------------")
    print("Route: ",route)
    print(context)
    print("--------------------------------------")

    ans = self.chain.invoke({"input_documents": [Document(context)], "human_input": query}, return_only_outputs=True)['output_text']
    return ans


class ConvMainWrapper:
  def __init__(self):
    self.conv_mains = {}
    self.univ_buffer = {}
    self.user_routes = None
    self.spl_model = SparseModel()

  def get_buffer(self,u_name):
    return self.conv_mains[u_name].memory.buffer

  def store_in_universal_buffer(self,u_name):
    tag_ans = run_symptom_tagger(self.conv_mains[u_name].memory.buffer[:512]) # Subject to change
    if tag_ans['is_symptom'].lower()=='yes':
      self.univ_buffer[u_name] = tag_ans['symptoms']
    self.user_routes = sem_route_runner(self.univ_buffer)

  def search_for_sim_pat(self,u_name):
    tag_ans = run_symptom_tagger(self.conv_mains[u_name].memory.buffer[:512])
    if tag_ans['is_symptom'].lower()=='yes':
      closest_match = self.user_routes(',  '.join(tag_ans['symptoms']))

    if closest_match.name is None:
      print("No Similar User Found")
    return closest_match.name

  def __call__(self,u_name,query,symp_image_path=None):
    if u_name not in self.conv_mains:
      self.conv_mains[u_name] = ConvMain()
    
    flag = len(self.get_buffer(u_name))==0
    res =  self.conv_mains[u_name](query, symp_image_path)
    
    if flag:
      if run_filter_tagger(query)['is_relevant'].lower()=='no':
        print(f"Warning: User {u_name} # SPAM Detected #")
      else:
        print(f"#OK#")
    return res




