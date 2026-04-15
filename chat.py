import os
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint, ChatHuggingFace
from langchain_community.vectorstores import FAISS
from langchain_classic.chains.retrieval import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

os.environ["HUGGINGFACEHUB_API_TOKEN"] = os.getenv("HUGGINGFACEHUB_API_TOKEN", "")
DB_FAISS_PATH = "vectorstore/db_faiss"

def chatbot():
    
    emb = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = FAISS.load_local(DB_FAISS_PATH, emb, allow_dangerous_deserialization=True)
    retriever = db.as_retriever(search_kwargs={'k': 3})

    llm = HuggingFaceEndpoint(
        repo_id="HuggingFaceH4/zephyr-7b-beta",
        temperature=0.3,
        max_new_tokens=512
    )
    
    model = ChatHuggingFace(llm=llm)

    system_prompt = (
        "You are Lexi, a supportive and knowledgeable legal companion for the people of India. "
    "Your goal is to help people understand their rights in a way that feels like talking to a well-informed, empathetic friend. "
    
    "GUIDELINES FOR YOUR TONE:"
    "1. Acknowledge: Start by acknowledging the user's situation or question. Use phrases like 'I understand that...', 'That's a very important question...', or 'I'm here to help you navigate this.' "
    "2. Be Conversational: Avoid bulleted lists if a paragraph feels more natural. Use human-like transitions like 'In simple terms,' or 'What this means for you is...' "
    "3. Stay Grounded: While being warm, never lose legal accuracy. If the context doesn't have the answer, say something like 'I've looked through the archives, but I don't have the specific details on that yet.' "
    "4. No Labels: Never use prefixes like 'Lexi:' or 'User:'. Just speak directly to the person. "
    
    
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])

    que_ans = create_stuff_documents_chain(model, prompt)
    rag = create_retrieval_chain(retriever, que_ans)
    
    return rag

if __name__ == "__main__":
    print(" Initializing Lexi Voice AI...")
    try:
        chain = chatbot()
        print(" Ready!")
        
        while True:
            user_query = input("\n Ask a legal question (or type 'exit' to quit): ")
            if user_query.lower() == 'exit':
                break
                
            print(" Lexi is searching the database...")
            
            response = chain.invoke({"input": user_query})
            
            print(f"\n Lexi says: {response['answer']}")
            
            print("\n Sources used:")
            for doc in response["context"]:
                print(f"- {doc.metadata.get('source', 'Unknown source')}")
                
    except Exception as e:
        print(f"\n An error occurred: {e}")