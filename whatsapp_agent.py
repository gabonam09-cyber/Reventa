"""
ANUBIS — Agente WhatsApp
Conecta tu DB de leads con Meta Graph API para enviar mensajes + fotos automáticamente.

Uso:
    python whatsapp_agent.py              # Procesa todos los leads pendientes
    python whatsapp_agent.py --watch      # Modo daemon: revisa cada 30s

Variables requeridas en .env:
    WHATSAPP_TOKEN        — API Key de Meta (Bearer token)
    WHATSAPP_PHONE_ID     — Phone Number ID de tu WhatsApp Business
    WHATSAPP_VERIFY_TOKEN — Token para el webhook (puedes poner cualquier string)
"""

import json
import sqlite3
import time
import argparse
import requests
from pathlib import Path
from dotenv import load_dotenv
import os

# ── Cargar .env ────────────────────────────────────────────────────────────────
load_dotenv(Path(__file__).with_name(".env"))

WHATSAPP_TOKEN    = os.getenv("WHATSAPP_TOKEN", "")
WHATSAPP_PHONE_ID = os.getenv("WHATSAPP_PHONE_ID", "")
DB_PATH = Path(__file__).parent.parent / "data" / "leads.sqlite"

# ── Catálogo de productos con fotos ───────────────────────────────────────────
# Cuando tengas las URLs de las fotos, sustitúyelas aquí
PRODUCT_CATALOG: dict[str, dict] = {
    "AirPods Pro 2": {
        "price": "$3,499",
        "photo_url": "",   # ← pega la URL pública de la foto aquí
        "discount": "10%",
        "msg": "Los AirPods Pro 2 que te interesaron tienen cancelación de ruido activa y audio espacial. 🔥",
    },
    "iPhone 15 Pro": {
        "price": "$18,999",
        "photo_url": "",
        "discount": "5%",
        "msg": "El iPhone 15 Pro que viste tiene cámara de 48MP y titanio. Viene sellado con garantía. 📱",
    },
    "Apple Watch S9": {
        "price": "$8,499",
        "photo_url": "",
        "discount": "8%",
        "msg": "El Apple Watch S9 que te gustó mide oxígeno, ECG y tiene GPS. Sellado y original. ⌚",
    },
    "Hoodie Oversized Negro": {
        "price": "$649",
        "photo_url": "",
        "discount": "15%",
        "msg": "La hoodie oversized que viste es talla única, algodón premium. Queda brutal. 🖤",
    },
    "Cargo Pants Caqui": {
        "price": "$799",
        "photo_url": "",
        "discount": "12%",
        "msg": "Los cargo pants caqui que te interesaron son de corte relajado, muy cómodos. 👖",
    },
    "Gorra Dad Hat Beige": {
        "price": "$299",
        "photo_url": "",
        "discount": "20%",
        "msg": "La dad hat beige que viste es ajustable y combina con todo. Un clásico. 🧢",
    },
    "Gorra Snapback Negra": {
        "price": "$349",
        "photo_url": "",
        "discount": "15%",
        "msg": "La snapback negra que te gustó tiene visera plana y talla única. 🧢",
    },
    "JBL Flip 6": {
        "price": "$1,899",
        "photo_url": "",
        "discount": "10%",
        "msg": "La JBL Flip 6 que viste es resistente al agua y tiene 12 hrs de batería. 🔊",
    },
    "Marshall Emberton II": {
        "price": "$2,499",
        "photo_url": "",
        "discount": "8%",
        "msg": "La Marshall Emberton II que te interesó tiene sonido brutal y batería de 30 hrs. 🎵",
    },
}

DEFAULT_PRODUCT = {
    "price": "",
    "photo_url": "",
    "discount": "10%",
    "msg": "Tenemos exactamente lo que buscas. Te mando los detalles ahora. 🔥",
}

# ── Meta Graph API ─────────────────────────────────────────────────────────────
GRAPH_URL = f"https://graph.facebook.com/v19.0/{WHATSAPP_PHONE_ID}/messages"
HEADERS = {
    "Authorization": f"Bearer {WHATSAPP_TOKEN}",
    "Content-Type": "application/json",
}


def send_text(phone: str, text: str) -> bool:
    """Envía un mensaje de texto por WhatsApp."""
    payload = {
        "messaging_product": "whatsapp",
        "to": f"52{phone}",
        "type": "text",
        "text": {"body": text},
    }
    r = requests.post(GRAPH_URL, headers=HEADERS, json=payload, timeout=10)
    return r.ok


def send_photo(phone: str, photo_url: str, caption: str) -> bool:
    """Envía una imagen con caption por WhatsApp."""
    payload = {
        "messaging_product": "whatsapp",
        "to": f"52{phone}",
        "type": "image",
        "image": {
            "link": photo_url,
            "caption": caption,
        },
    }
    r = requests.post(GRAPH_URL, headers=HEADERS, json=payload, timeout=10)
    return r.ok


def send_lead_sequence(lead: dict) -> bool:
    """
    Secuencia completa para un lead:
    1. Mensaje de bienvenida personalizado con descuento
    2. Foto del producto (si tiene URL)
    3. Mensaje de cierre invitando a responder
    """
    name    = lead.get("name") or "amigo"
    phone   = lead.get("company", "")   # guardamos phone en campo company
    product = lead.get("goal", "")
    first   = name.split()[0].capitalize()

    info = PRODUCT_CATALOG.get(product, DEFAULT_PRODUCT)
    discount = info["discount"]
    photo_url = info["photo_url"]
    detail_msg = info["msg"]
    price = info["price"]

    # 1. Mensaje de bienvenida
    welcome = (
        f"Hola {first} 👋 Soy de *ANUBIS*, vi que te interesó *{product}*.\n\n"
        f"Tenemos un descuento exclusivo *solo para ti hoy: {discount} OFF* "
        f"{'— quedaría en ' + price + ' 🔥' if price else '🔥'}\n\n"
        f"{detail_msg}"
    )
    ok1 = send_text(phone, welcome)
    if not ok1:
        return False

    time.sleep(1.5)

    # 2. Foto del producto (si hay URL)
    if photo_url:
        send_photo(
            phone,
            photo_url,
            caption=f"📦 {product} — {discount} OFF solo hoy para ti.",
        )
        time.sleep(1.5)

    # 3. Mensaje de cierre
    close = (
        "¿Te lo apartamos? Solo dime y coordinamos el pago y envío 🚀\n"
        "Enviamos a todo México, entrega en 24–48 hrs express."
    )
    ok3 = send_text(phone, close)
    return ok3


# ── Base de datos ──────────────────────────────────────────────────────────────
def get_pending_leads() -> list[dict]:
    """Obtiene leads que aún no han sido contactados por WhatsApp."""
    if not DB_PATH.exists():
        print(f"⚠️  DB no encontrada en {DB_PATH}")
        return []

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    # Intentamos con tabla 'leads', ajusta el nombre si es diferente
    try:
        cur.execute("""
            SELECT * FROM leads
            WHERE (wa_sent IS NULL OR wa_sent = 0)
            AND company != ''
            AND company IS NOT NULL
        """)
        rows = [dict(r) for r in cur.fetchall()]
    except sqlite3.OperationalError as e:
        print(f"⚠️  Error consultando DB: {e}")
        print("   Verifica el nombre de tu tabla de leads.")
        rows = []

    conn.close()
    return rows


def mark_as_sent(lead_id: int) -> None:
    """Marca el lead como contactado para no enviarle doble mensaje."""
    conn = sqlite3.connect(DB_PATH)
    try:
        conn.execute("UPDATE leads SET wa_sent = 1 WHERE id = ?", (lead_id,))
        conn.commit()
    except sqlite3.OperationalError:
        # Si no existe la columna, la creamos
        conn.execute("ALTER TABLE leads ADD COLUMN wa_sent INTEGER DEFAULT 0")
        conn.execute("UPDATE leads SET wa_sent = 1 WHERE id = ?", (lead_id,))
        conn.commit()
    conn.close()


# ── Main ───────────────────────────────────────────────────────────────────────
def process_leads() -> None:
    leads = get_pending_leads()
    if not leads:
        print("✓ No hay leads pendientes.")
        return

    print(f"📬 Procesando {len(leads)} lead(s)...")

    for lead in leads:
        phone = lead.get("company", "")
        name  = lead.get("name", "Sin nombre")

        if not phone or len(phone) < 10:
            print(f"  ⚠️  Lead {lead.get('id')} sin teléfono válido — omitiendo.")
            continue

        print(f"  → Enviando a {name} ({phone})...")
        ok = send_lead_sequence(lead)

        if ok:
            mark_as_sent(lead["id"])
            print(f"  ✅ Enviado a {name}")
        else:
            print(f"  ❌ Error enviando a {name} — revisa WHATSAPP_TOKEN y WHATSAPP_PHONE_ID")


def watch_mode(interval: int = 30) -> None:
    print(f"👁  Modo watch activo — revisando cada {interval}s. Ctrl+C para detener.\n")
    while True:
        process_leads()
        time.sleep(interval)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Agente WhatsApp ANUBIS")
    parser.add_argument("--watch", action="store_true", help="Modo daemon: revisa leads cada 30s")
    parser.add_argument("--interval", type=int, default=30, help="Segundos entre revisiones (default: 30)")
    args = parser.parse_args()

    if not WHATSAPP_TOKEN or not WHATSAPP_PHONE_ID:
        print("❌ Faltan variables en .env:")
        print("   WHATSAPP_TOKEN=tu_token_aqui")
        print("   WHATSAPP_PHONE_ID=tu_phone_id_aqui")
        exit(1)

    if args.watch:
        watch_mode(args.interval)
    else:
        process_leads()