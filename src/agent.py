import os
import json
import logging
from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
from langchain_google_genai import ChatGoogleGenerativeAI
from models import ProductEnrichment

class ProductEnrichmentAgent:
    def __init__(self, model_name="models/gemini-2.5-pro", temperature=0.2):
        # Load environment variables (API key etc.)
        load_dotenv()

        # Initialize logger
        self.logger = logging.getLogger(__name__)

        # Expects a valid JSON response.
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", "Sen bir ürün zeka asistanısın."),
            ("human", """
Aşağıdaki ürün açıklamasına göre şu alanları ingilizce çıkar ve yalnızca geçerli bir JSON objesi olarak döndür:

- category: Genel kategori (örnek: home decor, electronics, clothing)
- sub_category: Genel kategori altındaki spesifik ürün tipi (örnek: lamp, mug, mirror) 
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

        # Initialize LLM
        self.llm = ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=os.getenv("GOOGLE_API_KEY"),
            temperature=temperature
        )

    def enrich(self, description: str) -> ProductEnrichment:
        """
        Main enrichment method that sends a prompt to the LLM and parses the response
        """
        try:
            chain = self.prompt | self.llm
            result = chain.invoke({"description": description})
            content_str = result.content.strip()

            # Remove Markdown-style JSON wrapping if exists
            if content_str.startswith("```json"):
                content_str = content_str[len("```json"):].strip()
            if content_str.endswith("```"):
                content_str = content_str[:-3].strip()

            parsed = json.loads(content_str)
            self.logger.info("Successfully enriched description: %s", description)
            return ProductEnrichment(**parsed)

        except Exception as e:
            self.logger.error("Failed to enrich description: %s", description)
            self.logger.error("Exception: %s", str(e))
            self.logger.error("LLM response content: %s", content_str if 'content_str' in locals() else "[Empty]")

            # Return empty fallback if parsing fails
            return ProductEnrichment(
                category=None,
                usage_context=None,
                sub_category=None,
                price_segment=None,
                material_type=None,
                target_gender=None,
                target_age_group=None,
                tags=[]
            )

# Optional standalone function interface
agent_instance = ProductEnrichmentAgent()

def enrich_product_description(description: str) -> ProductEnrichment:
    return agent_instance.enrich(description)
