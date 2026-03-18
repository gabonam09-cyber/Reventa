"""
customer_sessions.py
====================
Maneja sesiones personalizadas por cliente en Anubis.

Cada vez que un customer visita o interactúa, se crea/actualiza su sesión
con su historial, preferencias y actividad. Todo guardado en SQLite.

Tablas nuevas que crea este módulo:
  - customer_sessions   → sesión activa por email
  - session_events      → historial de acciones (vistas, clics, compras)
  - customer_preferences → categorías favoritas, productos vistos

Uso rápido:
    from customer_sessions import SessionManager
    sm = SessionManager()
    session = sm.get_or_create("gabo.nam09@gmail.com", name="Gabriel")
    sm.track_event(session["session_id"], "view_product", {"product": "AirPods Pro 2"})
    prefs = sm.get_preferences(session["session_id"])
"""

import sqlite3
import uuid
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

# ── Config ──────────────────────────────────────────────────────────────────
DB_PATH = Path(__file__).parent / "data" / "leads.sqlite"
SESSION_TTL_HOURS = 24  # La sesión expira si no hay actividad en 24 hrs


# ── DB Setup ─────────────────────────────────────────────────────────────────
def get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row  # Resultados como dict
    return conn


def init_tables():
    """Crea las tablas de sesiones si no existen."""
    conn = get_connection()
    cursor = conn.cursor()

    # Sesiones activas
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS customer_sessions (
            session_id     TEXT PRIMARY KEY,
            email          TEXT NOT NULL,
            name           TEXT,
            created_at     TEXT NOT NULL,
            last_seen      TEXT NOT NULL,
            is_active      INTEGER DEFAULT 1,
            visit_count    INTEGER DEFAULT 1,
            lead_id        INTEGER REFERENCES leads(id)
        )
    """)

    # Historial de eventos por sesión
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS session_events (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id  TEXT NOT NULL REFERENCES customer_sessions(session_id),
            event_type  TEXT NOT NULL,
            payload     TEXT,  -- JSON con detalles del evento
            created_at  TEXT NOT NULL
        )
    """)

    # Preferencias del cliente (categorías vistas, productos, etc.)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS customer_preferences (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id      TEXT NOT NULL REFERENCES customer_sessions(session_id),
            email           TEXT NOT NULL,
            fav_category    TEXT,   -- apple | ropa | gorras | bocinas
            viewed_products TEXT,   -- JSON array de productos vistos
            cart_items      TEXT,   -- JSON array del carrito
            last_updated    TEXT NOT NULL
        )
    """)

    conn.commit()
    conn.close()
    print("✅ Tablas de sesiones listas.")


# ── Session Manager ───────────────────────────────────────────────────────────
class SessionManager:
    def __init__(self):
        init_tables()

    # ── Crear o recuperar sesión ──────────────────────────────────────────
    def get_or_create(self, email: str, name: Optional[str] = None) -> dict:
        """
        Busca si el email ya tiene una sesión activa (últimas 24 hrs).
        Si existe → la reutiliza y suma una visita.
        Si no → crea una nueva sesión.
        Retorna el dict de la sesión.
        """
        conn = get_connection()
        cursor = conn.cursor()
        now = datetime.utcnow().isoformat()
        cutoff = (datetime.utcnow() - timedelta(hours=SESSION_TTL_HOURS)).isoformat()

        # Buscar sesión activa reciente
        cursor.execute("""
            SELECT * FROM customer_sessions
            WHERE email = ? AND last_seen > ? AND is_active = 1
            ORDER BY last_seen DESC LIMIT 1
        """, (email, cutoff))
        row = cursor.fetchone()

        if row:
            # Sesión existente → actualizar last_seen y visit_count
            cursor.execute("""
                UPDATE customer_sessions
                SET last_seen = ?, visit_count = visit_count + 1, name = COALESCE(?, name)
                WHERE session_id = ?
            """, (now, name, row["session_id"]))
            conn.commit()
            session = dict(row)
            session["status"] = "returning"
            print(f"👋 Bienvenido de vuelta, {session.get('name') or email} (visita #{session['visit_count'] + 1})")
        else:
            # Nueva sesión
            session_id = str(uuid.uuid4())

            # Buscar si existe como lead
            cursor.execute("SELECT id FROM leads WHERE email = ? ORDER BY id DESC LIMIT 1", (email,))
            lead_row = cursor.fetchone()
            lead_id = lead_row["id"] if lead_row else None

            cursor.execute("""
                INSERT INTO customer_sessions (session_id, email, name, created_at, last_seen, lead_id)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (session_id, email, name, now, now, lead_id))

            # Preferencias vacías iniciales
            cursor.execute("""
                INSERT INTO customer_preferences (session_id, email, viewed_products, cart_items, last_updated)
                VALUES (?, ?, '[]', '[]', ?)
            """, (session_id, email, now))

            conn.commit()
            session = {
                "session_id": session_id,
                "email": email,
                "name": name,
                "created_at": now,
                "last_seen": now,
                "visit_count": 1,
                "lead_id": lead_id,
                "status": "new",
            }
            print(f"🆕 Nueva sesión creada para {name or email} → {session_id[:8]}...")

        conn.close()
        return session

    # ── Registrar evento ──────────────────────────────────────────────────
    def track_event(self, session_id: str, event_type: str, payload: Optional[dict] = None):
        """
        Registra cualquier acción del cliente.

        Tipos de evento sugeridos:
          - "view_product"    → payload: {"product": "AirPods Pro 2", "category": "apple"}
          - "add_to_cart"     → payload: {"product": "...", "price": "$3,499"}
          - "remove_from_cart"
          - "checkout_start"
          - "contact_whatsapp"
          - "filter_category" → payload: {"category": "ropa"}
          - "page_view"       → payload: {"page": "/"}
        """
        conn = get_connection()
        cursor = conn.cursor()
        now = datetime.utcnow().isoformat()

        cursor.execute("""
            INSERT INTO session_events (session_id, event_type, payload, created_at)
            VALUES (?, ?, ?, ?)
        """, (session_id, event_type, json.dumps(payload or {}), now))

        # Actualizar last_seen en la sesión
        cursor.execute("""
            UPDATE customer_sessions SET last_seen = ? WHERE session_id = ?
        """, (now, session_id))

        # Si es una vista de producto → actualizar viewed_products
        if event_type == "view_product" and payload and "product" in payload:
            self._add_viewed_product(cursor, session_id, payload["product"])

        # Si añade al carrito → actualizar cart_items
        if event_type == "add_to_cart" and payload:
            self._add_to_cart(cursor, session_id, payload)

        # Si filtra categoría → actualizar fav_category
        if event_type == "filter_category" and payload and "category" in payload:
            cursor.execute("""
                UPDATE customer_preferences SET fav_category = ?, last_updated = ?
                WHERE session_id = ?
            """, (payload["category"], now, session_id))

        conn.commit()
        conn.close()

    # ── Leer preferencias ─────────────────────────────────────────────────
    def get_preferences(self, session_id: str) -> dict:
        """Retorna preferencias actuales del cliente."""
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute("""
            SELECT cp.*, cs.name, cs.email, cs.visit_count
            FROM customer_preferences cp
            JOIN customer_sessions cs ON cp.session_id = cs.session_id
            WHERE cp.session_id = ?
        """, (session_id,))
        row = cursor.fetchone()
        conn.close()

        if not row:
            return {}

        prefs = dict(row)
        prefs["viewed_products"] = json.loads(prefs.get("viewed_products") or "[]")
        prefs["cart_items"] = json.loads(prefs.get("cart_items") or "[]")
        return prefs

    # ── Historial de eventos ──────────────────────────────────────────────
    def get_history(self, session_id: str, limit: int = 50) -> list:
        """Retorna los últimos N eventos de la sesión."""
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute("""
            SELECT event_type, payload, created_at
            FROM session_events
            WHERE session_id = ?
            ORDER BY created_at DESC LIMIT ?
        """, (session_id, limit))
        rows = cursor.fetchall()
        conn.close()
        return [
            {**dict(r), "payload": json.loads(r["payload"])}
            for r in rows
        ]

    # ── Resumen de un cliente ─────────────────────────────────────────────
    def get_customer_summary(self, email: str) -> dict:
        """
        Resumen completo de un cliente: todas sus sesiones, eventos y preferencias.
        Útil para el panel de admin o para personalizar mensajes de WhatsApp.
        """
        conn = get_connection()
        cursor = conn.cursor()

        cursor.execute("""
            SELECT cs.*, cp.fav_category, cp.viewed_products, cp.cart_items
            FROM customer_sessions cs
            LEFT JOIN customer_preferences cp ON cs.session_id = cp.session_id
            WHERE cs.email = ?
            ORDER BY cs.last_seen DESC
        """, (email,))
        sessions = [dict(r) for r in cursor.fetchall()]

        # Total de visitas
        total_visits = sum(s.get("visit_count", 1) for s in sessions)

        # Productos vistos en todas las sesiones
        all_viewed = []
        for s in sessions:
            viewed = json.loads(s.get("viewed_products") or "[]")
            all_viewed.extend(viewed)

        # Último lead registrado
        cursor.execute("""
            SELECT * FROM leads WHERE email = ? ORDER BY id DESC LIMIT 1
        """, (email,))
        lead = cursor.fetchone()
        conn.close()

        return {
            "email": email,
            "name": sessions[0].get("name") if sessions else None,
            "total_sessions": len(sessions),
            "total_visits": total_visits,
            "last_seen": sessions[0].get("last_seen") if sessions else None,
            "fav_category": sessions[0].get("fav_category") if sessions else None,
            "all_viewed_products": list(set(all_viewed)),
            "lead": dict(lead) if lead else None,
            "sessions": sessions,
        }

    # ── Cerrar sesión ─────────────────────────────────────────────────────
    def close_session(self, session_id: str):
        """Marca la sesión como inactiva."""
        conn = get_connection()
        conn.execute("""
            UPDATE customer_sessions SET is_active = 0 WHERE session_id = ?
        """, (session_id,))
        conn.commit()
        conn.close()
        print(f"🔒 Sesión {session_id[:8]}... cerrada.")

    # ── Ver todas las sesiones activas (admin) ────────────────────────────
    def list_active_sessions(self) -> list:
        """Lista todas las sesiones activas en las últimas 24 hrs."""
        conn = get_connection()
        cursor = conn.cursor()
        cutoff = (datetime.utcnow() - timedelta(hours=SESSION_TTL_HOURS)).isoformat()
        cursor.execute("""
            SELECT cs.session_id, cs.email, cs.name, cs.visit_count,
                   cs.last_seen, cp.fav_category
            FROM customer_sessions cs
            LEFT JOIN customer_preferences cp ON cs.session_id = cp.session_id
            WHERE cs.last_seen > ? AND cs.is_active = 1
            ORDER BY cs.last_seen DESC
        """, (cutoff,))
        rows = [dict(r) for r in cursor.fetchall()]
        conn.close()
        return rows

    # ── Helpers privados ──────────────────────────────────────────────────
    def _add_viewed_product(self, cursor, session_id: str, product_name: str):
        cursor.execute("""
            SELECT viewed_products FROM customer_preferences WHERE session_id = ?
        """, (session_id,))
        row = cursor.fetchone()
        if row:
            viewed = json.loads(row["viewed_products"] or "[]")
            if product_name not in viewed:
                viewed.append(product_name)
            cursor.execute("""
                UPDATE customer_preferences SET viewed_products = ?, last_updated = ?
                WHERE session_id = ?
            """, (json.dumps(viewed), datetime.utcnow().isoformat(), session_id))

    def _add_to_cart(self, cursor, session_id: str, item: dict):
        cursor.execute("""
            SELECT cart_items FROM customer_preferences WHERE session_id = ?
        """, (session_id,))
        row = cursor.fetchone()
        if row:
            cart = json.loads(row["cart_items"] or "[]")
            cart.append(item)
            cursor.execute("""
                UPDATE customer_preferences SET cart_items = ?, last_updated = ?
                WHERE session_id = ?
            """, (json.dumps(cart), datetime.utcnow().isoformat(), session_id))


# ── CLI rápido para probar ────────────────────────────────────────────────────
if __name__ == "__main__":
    sm = SessionManager()

    print("\n── TEST: Crear sesión nueva ──")
    s = sm.get_or_create("gabo.nam09@gmail.com", name="Gabriel")
    print(f"Session ID: {s['session_id'][:8]}... | Status: {s['status']}")

    print("\n── TEST: Registrar eventos ──")
    sid = s["session_id"]
    sm.track_event(sid, "page_view", {"page": "/"})
    sm.track_event(sid, "filter_category", {"category": "apple"})
    sm.track_event(sid, "view_product", {"product": "AirPods Pro 2", "category": "apple"})
    sm.track_event(sid, "add_to_cart", {"product": "AirPods Pro 2", "price": "$3,499"})
    sm.track_event(sid, "contact_whatsapp", {"product": "AirPods Pro 2"})
    print("✅ 5 eventos registrados")

    print("\n── TEST: Preferencias ──")
    prefs = sm.get_preferences(sid)
    print(f"Categoría favorita: {prefs.get('fav_category')}")
    print(f"Productos vistos: {prefs.get('viewed_products')}")
    print(f"Carrito: {prefs.get('cart_items')}")

    print("\n── TEST: Resumen del cliente ──")
    summary = sm.get_customer_summary("gabo.nam09@gmail.com")
    print(f"Total sesiones: {summary['total_sessions']}")
    print(f"Total visitas: {summary['total_visits']}")
    print(f"Último lead goal: {summary['lead']['goal'] if summary['lead'] else 'N/A'}")

    print("\n── TEST: Sesiones activas ──")
    active = sm.list_active_sessions()
    print(f"{len(active)} sesión(es) activa(s)")
    for a in active:
        print(f"  → {a['name'] or a['email']} | visitas: {a['visit_count']} | categoría: {a.get('fav_category', '-')}")