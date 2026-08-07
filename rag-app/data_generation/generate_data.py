from faker import Faker
import random
import json
from datetime import date, timedelta
import pandas as pd

from sqlalchemy import create_engine, text
from config import settings
from logging_config import get_logger
from dateutil.relativedelta import relativedelta

def mevsim_carpani(ay_numarasi: int) -> float:
    if ay_numarasi in (6, 7, 8):        # yaz -> tüketim artar
        return random.uniform(1.3, 1.6)
    elif ay_numarasi in (12, 1, 2):     # kış -> tüketim azalır
        return random.uniform(0.7, 0.9)
    else:                                # ilkbahar/sonbahar -> normal
        return random.uniform(0.95, 1.1)

def outlier_carpani() -> float:
    if random.random() < 0.03:   # %3 ihtimalle
        return random.uniform(3.0, 6.0)
    return 1.0    

IZMIR_ILCELERI = ["Konak", "Karşıyaka", "Bornova", "Buca", "Çiğli", "Gaziemir", "Bayraklı", "Güzelbahçe", "Narlıdere", "Balçova", "Karabağlar", "Menemen", "Aliağa", "Foça", "Seferihisar",
                   "Urla", "Dikili", "Bergama", "Kınık", "Ödemiş", "Tire", "Selçuk", "Menderes", "Torbalı", "Kemalpaşa", "Sasalı", "Karaburun", "Çeşme", "Beydağ", "Bayındır", "Kiraz", "Menemen", "Aliağa"]
MAHALLE_KOKLERI = ["Cumhuriyet", "Atatürk", "Barış", "Yeşiltepe", "Çamlık", "Fatih", "Yıldız", "Güzeltepe", "Hürriyet", "İstiklal", "Kurtuluş", "Sakarya", "Şehitler", "Bahçelievler", "Gazi", "Mevlana", "Mimar Sinan", "Namık Kemal", "Ortaköy", "Pınarbaşı"]
SOKAK_KOKLERI = ["Gül", "Menekşe", "Lale", "Zeytin", "Çınar", "Akasya", "Manolya", "Defne", "Kavak", "Sedir", "Karanfil", "Söğüt", "Ihlamur", "Kestane", "Mimoza", "Sarıyer", "Beyazıt", "Topkapı", "Fenerbahçe", "Kadıköy"]

fake = Faker("tr_TR")
Faker.seed(42)
random.seed(42)

musteriler = []
for i in range(500):
    musteriler.append({
        "ad": fake.first_name(),
        "soyad": fake.last_name(),
        "tc_no": fake.ssn(),
        "abonelik_no": f"AB-{i+1:06d}",
        "sayac_no": None if random.random() < 0.05 else f"SYC-{i+1:06d}", #yüzde 5 ihtimalle olmaması durumu
        "kayit_tarihi": fake.date_between(start_date="-5y", end_date="today"), #5 yıl öncesine kadar fake date
    })

df_musteriler = pd.DataFrame(musteriler)
df_musteriler.to_csv("musteriler.csv" , index = False, encoding="utf-8-sig")
print("musteriler.csv kaydedildi")

adresler = []
for i in range(500):
    adresler.append({
        "il": "İzmir",
        "ilce": random.choice(IZMIR_ILCELERI),
        "mahalle": random.choice(MAHALLE_KOKLERI) + " Mahallesi",
        "sokak_cadde": random.choice(SOKAK_KOKLERI)+ " " + random.choice(["Sokağı", "Caddesi"] ),
        "bina_no": str(random.randint(1, 150)),
        "daire_no": str(random.randint(1, 40)),
        "posta_kodu": str(random.randint(35000,35990)), # İzmir posta kodları 35 ile başlıyor
    })

su_tuketimi = []
for musteri in musteriler:
    ilk_endeks = round(random.uniform(100, 5000), 2)       #başlangıç sayaç değeri
    baslangic_tarihi = musteri["kayit_tarihi"]
    for ay in range(24): #iç döngü: o müşterinin 24 ayı
        okuma_tarihi = baslangic_tarihi + relativedelta(months=ay)  # her ayın aynı günü
        taban_tuketim = random.uniform(5.0, 25.0)
        tuketim = round(taban_tuketim * mevsim_carpani(okuma_tarihi.month) * outlier_carpani(), 2)
        son_endeks = round(ilk_endeks + tuketim, 2)  
        
    
        su_tuketimi.append({
          "ilk_endeks": ilk_endeks,
          "son_endeks": son_endeks,
          "tuketim_m3": tuketim,
          "okuma_tarihi": okuma_tarihi,
        })
        ilk_endeks=son_endeks


faturalar = []
for tuketim_kaydı in su_tuketimi:
    tuketim = tuketim_kaydı["tuketim_m3"]
    okuma_tarihi = tuketim_kaydı["okuma_tarihi"]
    birim_fiyat = 25.50  # TL/m3 -varsayım

    faturalar.append({
        "donem": okuma_tarihi.replace(day=1),          
        "tuketim_m3": tuketim,                          
        "tutar": round(tuketim * birim_fiyat, 2),        
        "son_odeme_tarihi": okuma_tarihi + timedelta(days=20),
        "odendi_mi": random.choices([True, False], weights=[75, 25])[0],  # BOOLEAN
    })
    
print(len(faturalar))



##################################################################


logger = get_logger(__name__)

engine = create_engine(settings.postgres_url)

with engine.begin() as conn:  # begin() = otomatik transaction, hata olursa hepsi geri alınır
    musteri_id_map = []  # index i -> gerçek musteri_id 

    # 1. musteriler'i INSERT et
    for m in musteriler:
        result = conn.execute(
            text("""
                INSERT INTO musteriler (ad, soyad, tc_no, abonelik_no, sayac_no, kayit_tarihi)
                VALUES (:ad, :soyad, :tc_no, :abonelik_no, :sayac_no, :kayit_tarihi)
                RETURNING musteri_id
            """),
            m  #sözlüğün key'leri otomatik eşleşiyor
        )
        musteri_id = result.scalar()  # RETURNING'den gelen tek değeri al
        musteri_id_map.append(musteri_id)

    logger.info("Musteriler eklendi: %d kayit", len(musteri_id_map))

    # 2. adresler INSERT 
    for i, a in enumerate(adresler):
        conn.execute(
            text("""
                INSERT INTO adresler (musteri_id, il, ilce, mahalle, sokak_cadde, bina_no, daire_no, posta_kodu)
                VALUES (:musteri_id, :il, :ilce, :mahalle, :sokak_cadde, :bina_no, :daire_no, :posta_kodu)
            """),
           {**a, "musteri_id": musteri_id_map[i]} ####en önemli satır, bağlama
        )

    logger.info("Adresler eklendi: %d kayit", len(adresler))

    # 3. su_tuketimi INSERT 
    tuketim_id_map = []  # index i -> gerçek tuketim_id
    for i, a in enumerate(su_tuketimi):
        musteri_index = i //24  #hangi musteri oldugunu anlamak icin
        result = conn.execute(
            text("""
                INSERT INTO su_tuketimi (musteri_id, ilk_endeks, son_endeks, tuketim_m3, okuma_tarihi)
                VALUES (:musteri_id, :ilk_endeks, :son_endeks, :tuketim_m3, :okuma_tarihi   )
                RETURNING tuketim_id
            """),
            {**a, "musteri_id": musteri_id_map[musteri_index]} ####en önemli satır, bağlama
         )
        tuketim_id = result.scalar()
        tuketim_id_map.append(tuketim_id)  #tuketim_id_map listesine ekleme

    logger.info("Su tüketimi eklendi: %d kayit", len(su_tuketimi))

    
    # 4. faturalar INSERT 
    for i, f in enumerate(faturalar):
        musteri_index = i//24 #hangi musteri oldugunu anlamak icin
        conn.execute(
            text("""
                INSERT INTO faturalar (musteri_id, tuketim_id, donem, tuketim_m3, tutar, son_odeme_tarihi, odendi_mi)
                VALUES (:musteri_id, :tuketim_id, :donem, :tuketim_m3, :tutar, :son_odeme_tarihi, :odendi_mi)
            """),
            {**f, "musteri_id": musteri_id_map[musteri_index], "tuketim_id": tuketim_id_map[i]  } ####en önemli satır, bağlama, i kalıyor cunku 12000 elemanlı ve su_tuketimi ile birebir aynı sırada
        )
    logger.info("Faturalar eklendi: %d kayit", len(faturalar))
