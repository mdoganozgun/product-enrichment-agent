import os
import json

from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
from langchain_google_genai import ChatGoogleGenerativeAI

from models import ProductEnrichment

load_dotenv()

prompt = ChatPromptTemplate.from_messages([
    ("system", "Sen bir ürün zeka asistanısın."),
    ("human", """
Aşağıdaki ürün açıklamasına göre şu alanları çıkar ve yalnızca geçerli bir JSON objesi olarak döndür:

- category: Genel kategori (örnek: home decor, electronics, clothing)
- usage_context: Nerede / nasıl kullanılır? (örnek: gift, kitchen, office)
- price_segment: "low", "mid" veya "high"
- material_type: Ürünün ana malzemesi (örnek: cotton, ceramic, metal)
- target_gender: "male", "female" veya "unisex"
- target_age_group: "children", "adults" veya "all ages"
- tags: 5–10 adet alakalı anahtar kelime (string listesi)

Her alan için yalnızca bir değer döndür.  
Eğer birden fazla olasılık varsa, en uygun olanı seç.  
Diğer olasılıkları 'tags' listesine ekle.

Sadece geçerli bir JSON objesi döndür, başka açıklama yapma.

Ürün açıklaması:
"{description}"
""")
])

llm = ChatGoogleGenerativeAI(
    model="models/gemini-2.5-pro",
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    temperature=0.2
)

# Ana fonksiyon
def enrich_product_description(description: str) -> ProductEnrichment:
    chain = prompt | llm
    result = chain.invoke({"description": description})

    # AIMessage içeriği alınır
    content_str = result.content.strip()

    # Eğer içerik ```json ile başlıyorsa temizle
    if content_str.startswith("```json"):
        content_str = content_str[len("```json"):].strip()
    if content_str.endswith("```"):
        content_str = content_str[:-3].strip()

    try:
        parsed = json.loads(content_str)
        return ProductEnrichment(**parsed)
    except Exception as e:
        print("JSON parse hatası:", e)
        print("Cevap:", content_str)
        return ProductEnrichment(
            category=None,
            usage_context=None,
            price_segment=None,
            material_type=None,
            target_gender=None,
            target_age_group=None,
            tags=[]
        )