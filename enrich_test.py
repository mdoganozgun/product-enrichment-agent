import os
import pandas as pd
from agent import enrich_product_description

df = pd.read_csv("data/Online Retail.csv")

descriptions = df["Description"].dropna().drop_duplicates().tolist()


if os.path.exists("data/enriched_cache.csv") and os.path.getsize("data/enriched_cache.csv") > 0:
    df_cache = pd.read_csv("data/enriched_cache.csv")
else:
    df_cache = pd.DataFrame()

# Zaten işlenmiş açıklamaları ayıkla
existing_desc = set(df_cache["Description"]) if not df_cache.empty else set()

# Yeni enrich edilecek açıklamaları belirle
descriptions_to_enrich = [desc for desc in descriptions if desc not in existing_desc]

# Yeni enrich edilenleri topla
new_rows = []
for desc in descriptions_to_enrich[:10]:
    try:
        enriched = enrich_product_description(desc)
        enriched_dict = enriched.model_dump() #Pydantic modelinden Python sözlüğüne çevirir !
        enriched_dict["Description"] = desc
        new_rows.append(enriched_dict)
    except Exception as e:
        print(f"Enrichment hatası: {desc} => {e}")

df_new = pd.DataFrame(new_rows)

# Cache ile birleştir ve kaydet
df_cache = pd.concat([df_cache, df_new], ignore_index=True)
df_cache = df_cache.drop_duplicates(subset=["Description"], keep="first")
df_cache.to_csv("data/enriched_cache.csv", index=False)

# Ana dataset ile enrich edilenleri birleştir
expected_cols = [
    "Description", "category", "sub_category","usage_context", "price_segment",
    "material_type", "target_gender", "target_age_group", "tags"
]
df_cache = df_cache[expected_cols]
df = df.merge(df_cache, on="Description", how="left")

# Description'a göre StockCode'lara da yay
columns_to_fill = [
    "category", "sub_category","usage_context", "price_segment",
    "material_type", "target_gender", "target_age_group", "tags"
]

for col in columns_to_fill:
    df[col] = df.groupby("Description")[col].transform(lambda x: x.ffill().bfill().infer_objects(copy=False))

# Bilgilendirme
print("Toplam satır:", len(df))
print("Enriched satır sayısı:", df['category'].notna().sum())
print(df[["Description", "category", "sub_category","usage_context", "tags"]].dropna().head(10))

# Sonuçları CSV'ye kaydet
df.to_csv("data/enriched_retail.csv", index=False)
print("Zenginleştirilmiş veri CSV'ye kaydedildi.")
