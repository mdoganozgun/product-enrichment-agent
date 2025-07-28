import pandas as pd
import os

retail_path = "data/enriched_retail.csv"
cache_path = "data/enriched_cache.csv"

# enriched_retail.csv dosyasını oku
df_retail = pd.read_csv(retail_path)

# Sadece enrichment yapılmış satırları filtrele
enriched_retail = df_retail[df_retail["category"].notna()].copy()

# enriched_cache.csv varsa oku, yoksa boş başlat
if os.path.exists(cache_path):
    df_cache = pd.read_csv(cache_path)
else:
    df_cache = pd.DataFrame(columns=enriched_retail.columns)

# Cache’te olmayan Description’ları bul
new_desc = ~enriched_retail["Description"].isin(df_cache["Description"])
df_new = enriched_retail[new_desc]

# Yeni enrichment'ları cache'e ekle
df_cache = pd.concat([df_cache, df_new], ignore_index=True)

# Her Description yalnızca bir kez kalsın (ilk enrich edilen tutulur)
df_cache = df_cache.drop_duplicates(subset="Description", keep="first")

# Güncellenmiş cache’i kaydet
df_cache.to_csv(cache_path, index=False)

print(f"Sync tamamlandı. Cache satır sayısı: {len(df_cache)}")