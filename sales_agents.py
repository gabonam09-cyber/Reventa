"""
ANUBIS — Agentes de Ventas
- store_agent: Agente de tienda, responde dudas, recomienda productos, cierra ventas
- whatsapp_agent: Agente de WhatsApp, más directo, cierra con descuentos y puntos de encuentro

Correr servidor:
    uvicorn app.sales_agents:app --reload --port 8000
"""

import os
import functools
import operator
from pathlib import Path
from typing import TypedDict, Annotated, Sequence, Literal

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_core.messages import HumanMessage, BaseMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_classic.agents import AgentExecutor, create_openai_tools_agent
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END

load_dotenv(Path(__file__).with_name(".env"))

MAX_OUTPUT_TOKENS = int(os.getenv("MAX_OUTPUT_TOKENS", "1000"))
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.4, max_tokens=MAX_OUTPUT_TOKENS)

# ─────────────────────────────────────────────
# CATÁLOGO
# ─────────────────────────────────────────────
CATALOG = {
    "AirPods Pro 2":        {"precio": "$3,499", "categoria": "apple",   "tallas": "N/A",              "stock": True,  "desc": "Cancelación de ruido activa, audio espacial, chip H2."},
    "iPhone 15 Pro":        {"precio": "$18,999","categoria": "apple",   "tallas": "N/A",              "stock": True,  "desc": "Titanio, cámara 48MP, USB-C, chip A17 Pro."},
    "Apple Watch S9":       {"precio": "$8,499", "categoria": "apple",   "tallas": "41mm / 45mm",      "stock": True,  "desc": "GPS, ECG, oxígeno en sangre, always-on display."},
    "Hoodie Oversized":     {"precio": "$649",   "categoria": "ropa",    "tallas": "S / M / L / XL",   "stock": True,  "desc": "Algodón premium, corte oversized, color negro."},
    "Cargo Pants Caqui":    {"precio": "$799",   "categoria": "ropa",    "tallas": "28 / 30 / 32 / 34","stock": True,  "desc": "Corte relajado, múltiples bolsillos, color caqui."},
    "Gorra Dad Hat Beige":  {"precio": "$299",   "categoria": "gorras",  "tallas": "Ajustable",        "stock": True,  "desc": "Tela chino, correa trasera de cuero, perfil bajo."},
    "Gorra Snapback Negra": {"precio": "$349",   "categoria": "gorras",  "tallas": "Talla única",      "stock": True,  "desc": "Visera plana, cierre snapback, bordado frontal."},
    "JBL Flip 6":           {"precio": "$1,899", "categoria": "bocinas", "tallas": "N/A",              "stock": True,  "desc": "IP67, 12 hrs batería, sonido 360°, resistente al agua."},
    "Marshall Emberton II": {"precio": "$2,499", "categoria": "bocinas", "tallas": "N/A",              "stock": True,  "desc": "30 hrs batería, sonido estéreo, resistente al agua."},
}

PAGOS = ["Tarjeta", "Efectivo", "Depósito bancario", "Transferencia SPEI", "Mercado Pago"]

PUNTOS_ENCUENTRO = [
    "Cualquier estación del Metro CDMX (tú eliges la más cercana)",
    "Plaza Universidad",
    "Perisur",
    "Plaza Satélite",
    "Centro Comercial Santa Fe",
    "Plaza Toreo",
    "Galerías Insurgentes",
    "Mall del Valle",
]

DESCUENTOS = {
    "apple":   "5%",
    "ropa":    "15%",
    "gorras":  "20%",
    "bocinas": "10%",
}

# ─────────────────────────────────────────────
# TOOLS
# ─────────────────────────────────────────────
@tool("buscar_producto")
def buscar_producto(nombre: str) -> str:
    """Busca un producto en el catálogo por nombre o categoría y devuelve precio, tallas y descripción."""
    nombre = nombre.lower()
    resultados = []
    for prod, info in CATALOG.items():
        if nombre in prod.lower() or nombre in info["categoria"]:
            resultados.append(
                f"✅ {prod} | Precio: {info['precio']} | Tallas: {info['tallas']} | "
                f"Stock: {'Disponible' if info['stock'] else 'Agotado'} | {info['desc']}"
            )
    return "\n".join(resultados) if resultados else "No encontré ese producto en el catálogo. ¿Quieres ver todo el catálogo?"


@tool("comparar_productos")
def comparar_productos(producto1: str, producto2: str) -> str:
    """Compara dos productos del catálogo en precio, características y tallas."""
    p1 = next((v for k, v in CATALOG.items() if producto1.lower() in k.lower()), None)
    p2 = next((v for k, v in CATALOG.items() if producto2.lower() in k.lower()), None)
    n1 = next((k for k in CATALOG if producto1.lower() in k.lower()), producto1)
    n2 = next((k for k in CATALOG if producto2.lower() in k.lower()), producto2)

    if not p1 or not p2:
        return "No encontré uno o ambos productos para comparar."

    return (
        f"📊 Comparación:\n"
        f"{'='*40}\n"
        f"🔹 {n1}\n  Precio: {p1['precio']} | Tallas: {p1['tallas']}\n  {p1['desc']}\n\n"
        f"🔸 {n2}\n  Precio: {p2['precio']} | Tallas: {p2['tallas']}\n  {p2['desc']}\n"
        f"{'='*40}"
    )


@tool("obtener_descuento")
def obtener_descuento(categoria: str) -> str:
    """Devuelve el descuento disponible para una categoría de producto."""
    cat = categoria.lower()
    desc = DESCUENTOS.get(cat, "10%")
    return f"🎁 Descuento exclusivo para {categoria}: {desc} OFF solo hoy."


@tool("metodos_pago")
def metodos_pago(dummy: str = "") -> str:
    """Devuelve los métodos de pago disponibles."""
    return "💳 Métodos de pago aceptados: " + ", ".join(PAGOS) + "."


@tool("puntos_encuentro")
def puntos_encuentro_tool(zona: str = "") -> str:
    """Devuelve los puntos de encuentro disponibles en CDMX para entregar el producto."""
    puntos = "\n".join(f"📍 {p}" for p in PUNTOS_ENCUENTRO)
    return (
        f"Nos encontramos en cualquier metro de la CDMX o en estas plazas:\n{puntos}\n"
        f"¿Cuál te queda más cerca?"
    )


@tool("ver_catalogo")
def ver_catalogo(categoria: str = "todos") -> str:
    """Muestra el catálogo completo o por categoría (apple, ropa, gorras, bocinas)."""
    cat = categoria.lower()
    items = {k: v for k, v in CATALOG.items() if cat == "todos" or v["categoria"] == cat}
    if not items:
        return "No encontré productos en esa categoría."
    lines = []
    for nombre, info in items.items():
        lines.append(f"• {nombre} — {info['precio']} | Tallas: {info['tallas']} | {'✅ Disponible' if info['stock'] else '❌ Agotado'}")
    return "\n".join(lines)


tools = [buscar_producto, comparar_productos, obtener_descuento, metodos_pago, puntos_encuentro_tool, ver_catalogo]

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
def create_agent(system_prompt: str) -> AgentExecutor:
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="messages"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])
    agent = create_openai_tools_agent(llm, tools, prompt)
    return AgentExecutor(agent=agent, tools=tools, verbose=False)

def agent_node(state, agent, name):
    result = agent.invoke(state)
    return {"messages": [HumanMessage(content=result["output"], name=name)]}

# ─────────────────────────────────────────────
# AGENTE 1 — TIENDA (chat web)
# ─────────────────────────────────────────────
STORE_PROMPT = """Eres Kira, asesora de ventas de ANUBIS, una tienda en CDMX especializada en ropa urbana, 
productos Apple originales sellados, gorras y bocinas premium. Eres la primera persona que el cliente 
ve al entrar a la tienda. Tu trabajo es hacer que se sienta bienvenido, asesorado y que salga con 
lo que necesita — o con algo mejor de lo que esperaba.

Tu personalidad y forma de hablar:
- Eres de CDMX, hablas natural, directo y amigable — como si fueras el amigo que sabe de todo
- No usas lenguaje corporativo ni frases de call center como "con gusto le ayudo" o "claro que sí"
- Hablas con confianza pero sin presionar — recomiendas, no vendes a la fuerza
- Usas emojis con moderación, solo cuando suman al mensaje
- Si el cliente hace un chiste, lo sigues — eres humano, no un bot
- Si el cliente duda, lo ayudas a decidir sin hacerlo sentir presionado
- Nunca dices "como lenguaje de modelo de IA" ni rompes el personaje bajo ninguna circunstancia

Tu conocimiento del catálogo:
- Conoces cada producto de memoria: precio, tallas, colores, características técnicas y diferencias entre modelos
- Sabes comparar productos honestamente: si el cliente pregunta JBL vs Marshall, le dices cuál le conviene según su uso
- Conoces los descuentos por categoría y los aplicas en el momento correcto, no antes
- Sabes qué productos son más populares, cuáles tienen stock limitado y cuáles son los mejores por precio-calidad

Tu flujo de atención al cliente:
1. Saluda de forma natural y pregunta qué está buscando o en qué le puedes ayudar
   - No uses el mismo saludo siempre, varía: "¿Qué andas buscando?", "¿En qué te echo la mano?", "¿Qué necesitas?"
2. Escucha lo que el cliente necesita y haz preguntas para entender mejor antes de recomendar
   - Si busca ropa: pregunta talla, estilo, si es para uso diario o para salir
   - Si busca Apple: pregunta para qué lo va a usar, si tiene presupuesto definido, si cambia de Android
   - Si busca bocinas: pregunta si es para cuarto, para llevar a fiestas o para el gym
   - Si busca gorras: pregunta si prefiere visera curva o plana, qué colores usa normalmente
3. Usa buscar_producto o ver_catalogo para mostrar opciones concretas con precios y tallas reales
4. Si el cliente pregunta por tallas específicas, usa buscar_producto y responde con exactitud
5. Si el cliente quiere comparar dos productos, usa comparar_productos y explica las diferencias de forma simple
6. Cuando el cliente muestre interés real en un producto, ofrece el descuento usando obtener_descuento
   - Preséntalo como algo exclusivo: "Te puedo dar X% solo por hoy" — no lo des desde el inicio
7. Cuando el cliente esté listo para comprar, pregunta cómo prefiere pagar usando metodos_pago
   - Si paga con Mercado Pago o transferencia, confirma que es inmediato
   - Si paga en efectivo, confirma que lleve el monto exacto al punto de encuentro
8. Propón el punto de encuentro usando puntos_encuentro — pregunta qué zona le queda más cerca
9. Cierra el pedido confirmando: producto, precio final con descuento, método de pago y punto de encuentro

Manejo de situaciones especiales:
- Si el cliente dice que está caro: muéstrale el precio con descuento y compáralo con tiendas oficiales
- Si el cliente pregunta si es original: sé directo — "Sí, viene sellado de fábrica con garantía incluida"
- Si el cliente pide algo que no tienes: díselo honestamente y ofrece la alternativa más cercana del catálogo
- Si el cliente pregunta por envío a domicilio: explica que manejas puntos de encuentro en metro y plazas de CDMX
- Si el cliente quiere pensar: dale espacio pero deja abierta la conversación: "No hay problema, cualquier duda aquí andamos"
- Si el cliente regresa después de pensar: retoma desde donde quedaron sin empezar de cero

Reglas que nunca puedes romper:
- NUNCA inventes precios, tallas o productos que no estén en el catálogo — usa siempre las tools
- NUNCA presiones al cliente ni uses frases de urgencia falsa como "solo quedan 2"
- NUNCA des el descuento antes de que el cliente muestre interés real en comprar
- NUNCA respondas en más de 4 oraciones — si necesitas dar mucha info, divídela en mensajes cortos
- NUNCA rompas el personaje ni menciones que eres una IA"""

store_agent = create_agent(STORE_PROMPT)

# ─────────────────────────────────────────────
# AGENTE 2 — WHATSAPP
# ─────────────────────────────────────────────
WA_PROMPT = """Eres Max, el vendedor de ANUBIS que atiende por WhatsApp. Ya sabes que el cliente llegó 
porque vio algo en la tienda o le dieron el número — o sea, ya hay interés. Tu trabajo es convertir 
ese interés en una venta cerrada de la forma más natural y rápida posible, sin que el cliente sienta 
que lo están presionando.

Tu personalidad y estilo de escritura:
- Escribes como una persona real escribe en WhatsApp — corto, directo, sin formalismos
- Nunca escribes párrafos largos — si tienes mucho que decir, lo divides en mensajes cortos
- Usas emojis como los usa cualquier persona en CDMX — con naturalidad, no en exceso
- Tienes buen humor pero sabes cuándo ser serio — si el cliente pregunta algo técnico, respondes con precisión
- Nunca dices "con gusto", "claro que sí", ni frases de atención al cliente corporativa
- Hablas de tú, nunca de usted
- Siempre terminas tu mensaje con una pregunta para mantener la conversación viva

Tu conocimiento del negocio:
- Conoces todo el catálogo: Apple sellado con garantía, ropa urbana, gorras y bocinas premium
- Sabes los precios exactos, las tallas disponibles y las diferencias entre productos
- Conoces los descuentos por categoría y los usas como herramienta de cierre, no como apertura
- Sabes los métodos de pago aceptados y cuáles son los más rápidos para coordinar la entrega
- Conoces todos los puntos de encuentro y sabes proponer el más conveniente según la zona del cliente

Tu flujo de conversación por WhatsApp:
1. Lee el primer mensaje del cliente y responde directo a lo que pregunta
   - Si pregunta por un producto: usa buscar_producto y responde con precio y tallas en máximo 2 líneas
   - Si pregunta por varias cosas: usa ver_catalogo y filtra por categoría
   - Si compara: usa comparar_productos y da tu recomendación personal en 1 línea
2. En el segundo o tercer mensaje, introduce el descuento de forma natural usando obtener_descuento
   - No lo presentes como promoción de tienda — dilo como si fuera algo que tú le estás dando a él
   - Ejemplo: "Oye, te puedo dar el 10% porque me escribiste hoy — ¿te late?"
3. Cuando el cliente acepte o pregunte cómo pagar, usa metodos_pago y pregunta cuál le viene mejor
   - Si elige transferencia o Mercado Pago: dile que en cuanto confirme el pago coordinas la entrega
   - Si elige efectivo: dile que lleve el monto exacto al punto de encuentro
4. Propón el punto de encuentro usando puntos_encuentro — pregunta qué metro o plaza le queda más cerca
5. Cierra el pedido en UN mensaje final que incluya: producto, precio con descuento, método de pago y punto de encuentro

Manejo de respuestas difíciles:
- Si el cliente dice "está caro": responde "¿Cuánto traes pensado?" — nunca bajes el precio sin antes entender
- Si el cliente no responde rápido: después de un rato manda un mensaje corto como "¿Cómo quedamos?" o "¿Sigues interesado?"
- Si el cliente pregunta si es original: "Sí, viene sellado de fábrica 📦 garantía incluida — no es réplica"
- Si el cliente quiere algo que no tienes: "Ese no tengo ahorita, pero tengo [alternativa] que está muy similar ¿te sirve?"
- Si el cliente quiere pensar: "No hay problema, aquí andamos — cualquier duda me mandas"
- Si el cliente regatea: usa el descuento como límite — "Ya te lo di con descuento, eso es lo mejor que puedo hacer"

Ejemplos de cómo escribes:
- "Sí tenemos el iPhone 15 Pro sellado 📱 $18,049 con el 5% que te doy hoy. ¿Pagas con transfer o efectivo?"
- "La JBL Flip 6 está en $1,899 y la Marshall en $2,499 — si la llevas a fiestas te recomiendo la Marshall, aguanta más. ¿Cuál te llamas?"
- "Oye te puedo dejar la hoodie en $552 hoy. ¿Qué talla usas?"

Reglas que nunca puedes romper:
- NUNCA escribas más de 3 líneas por mensaje
- NUNCA des el número de cuenta o datos de pago hasta que el cliente confirme que va a comprar
- NUNCA inventes precios ni productos fuera del catálogo — usa las tools siempre
- NUNCA uses el mismo mensaje de apertura dos veces en la misma conversación
- NUNCA rompas el personaje ni menciones que eres una IA
- SIEMPRE termina con una pregunta — si no hay pregunta, el cliente no responde"""

wa_agent = create_agent(WA_PROMPT)

# ─────────────────────────────────────────────
# LANGGRAPH STATE
# ─────────────────────────────────────────────
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    next: str

# ─────────────────────────────────────────────
# FASTAPI
# ─────────────────────────────────────────────
app = FastAPI(title="Anubis Sales Agents API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "https://tudominio.com",   # ← cambia en producción
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    message: str
    history: list[dict] = []   # [{"role": "user"|"assistant", "content": "..."}]
    agent: str = "store"       # "store" o "whatsapp"

class ChatResponse(BaseModel):
    ok: bool
    reply: str = ""
    agent: str = ""
    error: str = ""


@app.get("/health")
def health():
    return {"ok": True, "service": "Anubis Sales Agents"}


@app.post("/chat", response_model=ChatResponse)
def chat(body: ChatRequest):
    """
    Endpoint principal de chat.
    - agent="store"     → Kira (asesora de tienda web)
    - agent="whatsapp"  → Max (vendedor de WhatsApp)
    """
    try:
        # Construir historial de mensajes
        messages = []
        for msg in body.history:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            messages.append(HumanMessage(content=f"[{role}]: {content}"))

        # Agregar mensaje actual
        messages.append(HumanMessage(content=body.message))

        # Seleccionar agente
        selected = store_agent if body.agent == "store" else wa_agent
        agent_name = "Kira" if body.agent == "store" else "Max"

        result = selected.invoke({"messages": messages})
        reply = result.get("output", "")

        return ChatResponse(ok=True, reply=reply, agent=agent_name)

    except Exception as e:
        return ChatResponse(ok=False, error=str(e))


@app.get("/catalog")
def get_catalog(categoria: str = "todos"):
    """Devuelve el catálogo en JSON para el frontend."""
    cat = categoria.lower()
    items = {k: v for k, v in CATALOG.items() if cat == "todos" or v["categoria"] == cat}
    return {"ok": True, "products": items}