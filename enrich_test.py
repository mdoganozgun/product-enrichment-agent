import pandas as pd

df = pd.read_csv("data/Online Retail.csv")

from agent import enrich_product_description

# Sadece açıklaması olan ve tekrar edenleri temizle
sample_descriptions = df["Description"].dropna().drop_duplicates().tolist()[:10]

results = []
for desc in sample_descriptions:
    enriched = enrich_product_description(desc)
    enriched_dict = enriched.model_dump()
    enriched_dict["Description"] = desc  # eşleştirme için description'ı ekle
    results.append(enriched_dict)

# Enriched DataFrame oluştur
df_enriched = pd.DataFrame(results)


# enriched bilgileri Description üzerinden df'ye eşle ve StockCode eşleşen tüm satırlara uygula
# Önce Description üzerinden birleştir
df = df.merge(df_enriched, on="Description", how="left")

# Her Description’a karşılık gelen StockCode bilgisini yakalayalım
# Sonra o StockCode’a sahip diğer satırlara da aynı enrichment bilgisini yay
columns_to_fill = ["category", "usage_context", "price_segment", "material_type", "target_gender", "target_age_group", "tags"]

for col in columns_to_fill:
    df[col] = df.groupby("Description")[col].transform(lambda x: x.ffill().bfill().infer_objects(copy=False))

print("Toplam satır:", len(df))
print("Enriched satır sayısı:", df["category"].notna().sum())
print(df[["Description", "category", "usage_context", "tags"]].dropna().head(10))
df.to_csv("data/enriched_retail.csv", index=False)
print("CSV'ye kaydedildi.")
