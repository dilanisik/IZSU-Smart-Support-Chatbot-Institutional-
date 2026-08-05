-- ============================================================
-- Hibrit RAG Uygulamasi -- Gun 2: Veritabani kurulum scripti
-- (Gun 1 semasindaki revizyona gore guncellenmistir)
-- Bu dosya PostgreSQL konteyneri ilk kez ayaga kalktiginda
-- Docker tarafindan otomatik calistirilir.
-- ============================================================

-- Musteri tablosu
CREATE TABLE musteriler (
    musteri_id      SERIAL PRIMARY KEY,
    ad              VARCHAR(50) NOT NULL,
    soyad           VARCHAR(50) NOT NULL,
    tc_no           VARCHAR(11) UNIQUE,          -- sentetik, gercek degil
    abonelik_no     VARCHAR(20) UNIQUE NOT NULL,
    sayac_no        VARCHAR(20) UNIQUE NOT NULL,
    kayit_tarihi    DATE NOT NULL DEFAULT CURRENT_DATE
);

-- Adres tablosu (bir musterinin birden fazla adresi olabilir -> 1:N)
CREATE TABLE adresler (
    adres_id        SERIAL PRIMARY KEY,
    musteri_id      INT NOT NULL REFERENCES musteriler(musteri_id),
    il              VARCHAR(50) NOT NULL,
    ilce            VARCHAR(50) NOT NULL,
    mahalle         VARCHAR(100),
    sokak_cadde     VARCHAR(150),
    bina_no         VARCHAR(10),
    daire_no        VARCHAR(10),
    posta_kodu      VARCHAR(10)
);

CREATE INDEX idx_adresler_musteri ON adresler (musteri_id);

-- Su tuketim kayitlari (sayac okumalari)
CREATE TABLE su_tuketimi (
    tuketim_id      SERIAL PRIMARY KEY,
    musteri_id      INT NOT NULL REFERENCES musteriler(musteri_id),
    ilk_endeks      NUMERIC(12,2) NOT NULL,
    son_endeks      NUMERIC(12,2) NOT NULL,
    tuketim_m3      NUMERIC(10,2) NOT NULL,
    okuma_tarihi    DATE NOT NULL,
    UNIQUE (musteri_id, okuma_tarihi),
    CONSTRAINT chk_endeks   CHECK (son_endeks > ilk_endeks),
    CONSTRAINT chk_tuketim  CHECK (tuketim_m3 = son_endeks - ilk_endeks)
);

CREATE INDEX idx_su_tuketimi_musteri ON su_tuketimi (musteri_id);
CREATE INDEX idx_su_tuketimi_tarih   ON su_tuketimi (okuma_tarihi);

-- Fatura kayitlari
CREATE TABLE faturalar (
    fatura_id           SERIAL PRIMARY KEY,
    musteri_id          INT NOT NULL REFERENCES musteriler(musteri_id),
    tuketim_id           INT NOT NULL REFERENCES su_tuketimi(tuketim_id),
    donem               DATE NOT NULL,
    tuketim_m3           NUMERIC(10,2) NOT NULL,
    tutar               NUMERIC(10,2) NOT NULL,
    son_odeme_tarihi     DATE NOT NULL,
    odendi_mi            BOOLEAN NOT NULL DEFAULT FALSE
);

CREATE INDEX idx_faturalar_musteri_donem ON faturalar (musteri_id, donem DESC);
CREATE INDEX idx_musteriler_ad_soyad     ON musteriler (ad, soyad);

