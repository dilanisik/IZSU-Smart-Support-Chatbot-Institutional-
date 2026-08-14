"""
retrieval/debug_ara.py

Gün 11 — Retrieval tanı aracı (LLM'e hiç gitmeden)

Amaç: bir sorunun Qdrant'tan gerçekte hangi chunk'ları getirdiğini, hangi skorla
ve TAM METNİYLE görmek — Ollama'yı beklemeden (5-10 dakika değil, birkaç saniye
sürer). Bu, retrieval hatası (madde #1-2: retrieval ≠ generation) ile generation
hatasını ayırt etmek için kullanılır:

  - Hiç chunk gelmiyorsa (esik altı) -> retrieval sorunu, embedding/eşik/veri sorunu.
  - Chunk geliyor ama LLM "bilgi yok" diyorsa -> ya chunk gerçekten alakasız
    (yanlış pozitif eşleşme) ya da model fazla temkinli; TAM METNİ görüp karar ver.

Kullanım (rag-app kök dizininden, venv aktifken):
    python -m retrieval.debug_ara "Son yıllarda su kayıp oranı yüzde kaçtır?"

    # Eşiği geçici olarak düşürüp daha fazla aday görmek için:
    python -m retrieval.debug_ara "Son yıllarda su kayıp oranı yüzde kaçtır?" --esik 0.3
"""

import argparse

from retrieval.rag_chain import soru_embedle, qdranttan_ara


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("soru", type=str)
    parser.add_argument("--esik", type=float, default=None, help="Belirtilmezse eşiksiz, top-k'nın tamamı gösterilir.")
    parser.add_argument("--top-k", type=int, default=8)
    args = parser.parse_args()

    print(f"\nSoru: {args.soru}\n{'=' * 70}")

    dense_vec, sparse_vec = soru_embedle(args.soru)
    adaylar = qdranttan_ara(dense_vec, sparse_vec, top_k=args.top_k)

    if not adaylar:
        print("Qdrant'tan hiç aday dönmedi (koleksiyon boş olabilir mi? kontrol et).")
        return

    for i, p in enumerate(adaylar, start=1):
        pl = p.payload
        esik_uzeri = "✅ eşik üstü" if (args.esik is None or p.score >= args.esik) else "❌ eşik altı"
        gecerli_mi = "✅ eşik üstü" if p.score >= 0.50 else "❌ (varsayılan eşik 0.50 altında)"
        print(f"\n[{i}] skor={p.score:.4f}  {gecerli_mi}")
        print(f"    dosya: {pl['dosya_adi']}  sayfa: {pl['sayfa_no']}  kategori: {pl['kategori']}")
        print(f"    metin: {pl['metin'][:400]}{'...' if len(pl['metin']) > 400 else ''}")

    print(f"\n{'=' * 70}")
    print(f"Toplam {len(adaylar)} aday gösterildi (top-{args.top_k}).")


if __name__ == "__main__":
    main()
