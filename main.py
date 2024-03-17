from common_utils import *
from utils import *

def run():
    print("Hello World")
    conv_main_wrapper = ConvMainWrapper()
    print("ConvMainWrapper created")
    conv_main_wrapper("user1","What is the medicine of Dengue")
    conv_main_wrapper("user2","Hi how are you ?")
    conv_main_wrapper("user3","Is it a case of serious blood loss? ", "/content/images.jpeg")

if __name__ == "__main__":
    os.environ["OPENAI_API"]=getpass("Enter the OpenAI API Key: ")
    print("Hello World")
    run()