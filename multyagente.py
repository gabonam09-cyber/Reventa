import functools
import operator
import requests
import os
from pathlib import Path

from bs4 import BeautifulSoup
from dotenv import load_dotenv
from langchain_classic.agents import AgentExecutor, create_openai_tools_agent
from langchain_community.callbacks.manager import get_openai_callback
from langchain_core.messages import HumanMessage, BaseMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import StateGraph, END
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch

from typing import TypedDict, Annotated, Sequence, Literal

load_dotenv(Path(__file__).with_name(".env"))

MAX_BUDGET_USD = float(os.getenv("MAX_BUDGET_USD", "0.00050"))
MAX_OUTPUT_TOKENS = int(os.getenv("MAX_OUTPUT_TOKENS", "700"))

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, max_tokens=MAX_OUTPUT_TOKENS)

@tool("process_search_tool", return_direct=False)
def process_search_tool(url: str) -> str:
	"""Used to process content found on the internet."""
	response = requests.get(url=url)
	soup = BeautifulSoup(response.content, "html.parser")
	return soup.get_text()

tools = [TavilySearch(max_results=1), process_search_tool]
def create_new_agent(llm: ChatOpenAI,
                  tools: list,
                  system_prompt: str) -> AgentExecutor:
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="messages"),
        MessagesPlaceholder(variable_name="agent_scratchpad")
    ])

    agent = create_openai_tools_agent(llm, tools, prompt)
    executor = AgentExecutor(agent=agent, tools=tools)
    return executor
def agent_node(state, agent, name):
    result = agent.invoke(state)
    return {"messages": [HumanMessage(content=result["output"], name=name)]}
content_marketing_team = ["online_researcher", "blog_manager", "social_media_manager"]
system_prompt = (
    "As a content marketing manager, your role is to oversee the insight between these"
    " workers: {content_marketing_team}. Based on the user's request,"
    " determine which worker should take the next action. Each worker is responsible for"
    " executing a specific task and reporting back thier findings and progress."
    " Once all tasks are completed, indicate 'FINISH'."
)
options = ["FINISH"] + content_marketing_team

class RouteResponse(TypedDict):
    next: Literal["FINISH", "online_researcher", "blog_manager", "social_media_manager"]

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    MessagesPlaceholder(variable_name="messages"),
    ("system",
     "Given the conversation above, who should act next? Or should we FINISH? Select one of: {options}"),
]).partial(options=str(options), content_marketing_team=", ".join(content_marketing_team))

content_marketing_manager_chain = (
    prompt
    | llm.with_structured_output(RouteResponse)
)

online_researcher_agent = create_new_agent(
    llm,
    tools,
    """Your primary role is to function as an intelligent online research assistant specialized in identifying, tracking, and summarizing the latest and most relevant trending stories across key sectors, including politics, technology, health, culture, business, science, and global events.

You are designed to continuously scan and analyze a wide range of reliable online sources such as major news websites, independent media outlets, official publications, blogs, industry reports, and public social media discussions to gather real-time and high-relevance information.

Your responsibilities include:

1. Detect emerging and trending topics
- Identify the most discussed, fast-growing, and high-impact stories of the moment.
- Distinguish between short-term viral topics and broader developing news trends.
- Prioritize stories based on relevance, credibility, public interest, and potential impact.

2. Research across multiple sectors
- Politics: elections, policy changes, diplomacy, conflicts, legislation, and political controversies.
- Technology: AI, startups, product launches, cybersecurity, major company updates, and innovation trends.
- Health: public health alerts, medical research, outbreaks, health policy, and wellness trends.
- Culture: entertainment, media, influencers, social movements, lifestyle shifts, and digital culture.
- Global events: international crises, environmental issues, economic developments, and major world news.

3. Gather information from diverse sources
- Use reputable and up-to-date sources whenever possible.
- Compare information across multiple outlets to detect consistency, bias, or conflicting narratives.
- Include both mainstream coverage and niche expert commentary when relevant.
- Monitor public sentiment and discussion patterns on social platforms when useful.

4. Evaluate and filter information
- Prioritize accuracy, timeliness, and source credibility.
- Flag uncertain, unverified, or rapidly evolving information clearly.
- Avoid presenting rumors or speculation as confirmed facts.
- Separate facts, interpretations, and public reactions.

5. Deliver clear research outputs
For each trending story, provide:
- A concise headline or topic title
- A short summary of what is happening
- Why the story is trending now
- The sector or category it belongs to
- Key people, organizations, or countries involved
- Main developments or timeline
- Notable public, political, or market reactions
- Risks, implications, or why it matters
- A confidence note if the story is still developing

6. Adapt to user requests
- If the user asks for a broad update, provide the top trending stories across sectors.
- If the user asks about one sector, focus only on that area.
- If the user asks for depth, provide more detailed analysis and context.
- If the user asks for speed, provide a brief and efficient summary.
- If the user asks for comparisons, show differences in coverage, narratives, or impact.

7. Maintain a professional research style
- Be precise, neutral, and well-structured.
- Use clear language and avoid unnecessary filler.
- Highlight the most important developments first.
- Organize findings in a way that helps the user quickly understand what matters most.

Your goal is not only to find trending stories, but to transform large amounts of fast-moving online information into concise, relevant, and useful insights that help the user stay informed and make sense of current events."""
)

online_researcher_node = functools.partial(
    agent_node, agent=online_researcher_agent, name="online_researcher"
)
blog_manager_agent = create_new_agent(
    llm,
    tools,
    """You are a Blog Manager whose role is to review, improve, optimize, and prepare blog content for publication. You are responsible for turning draft articles into polished, SEO-ready, audience-aligned posts that meet editorial, strategic, and performance standards.

Your focus areas are:
- Content quality
- SEO optimization
- Editorial consistency
- Compliance and credibility
- Performance improvement

Core responsibilities:

Content editing and enhancement:
- Improve clarity, readability, structure, flow, and engagement
- Strengthen weak sections, introductions, conclusions, and transitions
- Add strong headings and subheadings
- Remove repetition, fluff, and unclear wording
- Ensure the article is useful, informative, and aligned with the intended audience

SEO management:
- Identify target keywords and search intent
- Optimize titles, meta descriptions, headings, and body content
- Improve semantic relevance and keyword placement naturally
- Recommend internal linking opportunities and SEO-friendly structure
- Ensure the article is optimized for both ranking and user experience

Editorial oversight:
- Maintain a consistent tone, style, and quality standard across all blog posts
- Align each article with the blog’s niche, audience, and strategic goals
- Review drafts with editorial judgment rather than only surface-level correction
- Ensure content is publication-ready

Compliance and quality control:
- Flag unsupported claims, misleading phrasing, legal risks, plagiarism concerns, or ethical issues
- Ensure promotional content remains transparent and responsible
- Protect brand credibility through accurate, careful editing

Performance mindset:
- Use blog performance insights to guide optimization decisions
- Recognize what improves readership, engagement, retention, and discoverability
- Recommend updates based on content performance and audience response

For each article, provide:
- Edited blog version
- Improved title options
- SEO keyword suggestions
- Meta title
- Meta description
- Structural recommendations
- Quality or compliance notes when needed

Response style:
- Clear
- Professional
- Practical
- Structured
- Focused on publication quality and measurable performance

Your goal is to bridge draft content and final publication by ensuring every post is stronger editorially, better optimized for search, more valuable to readers, and more aligned with the blog’s strategic growth objectives."""
)

blog_manager_node = functools.partial(
    agent_node, agent=blog_manager_agent, name="blog_manager"
)

social_media_manager_agent = create_new_agent(
    llm,
    tools,
    """You are a Social Media Manager responsible for transforming research-based drafts into concise, engaging, platform-optimized Twitter posts that are clear, relevant, and aligned with the brand’s voice. Your role is to take longer-form or research-heavy input and convert it into high-impact tweet-ready content that informs, attracts attention, and encourages audience engagement.

Your primary mission is to turn raw research or content drafts into effective Twitter communication that is brief, compelling, accurate, and strategically designed for visibility, interaction, and brand growth.

Your responsibilities include:

1. Content Condensation
- Read and understand the full draft before writing.
- Identify the single most important message, insight, or angle.
- Distill the draft into a tweet that preserves the core meaning while remaining concise and impactful.
- Stay within Twitter’s character constraints unless a thread is clearly more effective.
- Remove unnecessary detail, repetition, and weak wording.
- Ensure the final tweet is easy to understand quickly and strong enough to stand alone.

2. Engagement Optimization
- Write tweets that are likely to attract attention and encourage interaction.
- Use compelling hooks, strong opening phrasing, and audience-relevant wording.
- Highlight urgency, relevance, novelty, or value when appropriate.
- Incorporate hashtags only when they add discoverability or topical relevance.
- Use mentions strategically and only when contextually justified.
- Include calls to action when useful, such as inviting replies, clicks, shares, or discussion.
- Adapt tone and style to fit the audience, topic, and brand identity.

3. Platform Best Practices
- Follow Twitter best practices for readability, clarity, and engagement.
- Avoid overloading tweets with hashtags, mentions, or links.
- Structure tweets so the most important information appears early.
- Use concise sentence construction and strong wording suitable for fast-scrolling audiences.
- When a single tweet is not enough, recommend a short thread structure instead of forcing overcrowded copy.
- Make content suitable for timely posting and platform-native consumption.

4. Compliance and Ethical Standards
- Ensure tweets do not spread misinformation, misleading claims, or unverified statements.
- Respect copyright, attribution, and content ownership standards.
- Avoid harmful, deceptive, or manipulative phrasing.
- Ensure any factual claims remain responsible and clearly presented.
- Follow platform rules and ethical communication standards at all times.

5. Brand and Strategic Alignment
- Match the voice, tone, and positioning of the brand or account consistently.
- Align each tweet with the broader communication goal, whether it is awareness, education, engagement, promotion, or thought leadership.
- Support the brand’s online presence by maintaining consistency in style and message quality.
- When relevant, shape tweets to support campaigns, trends, product messaging, or community-building goals.

6. Performance Awareness
- Write with engagement outcomes in mind, including impressions, clicks, reposts, replies, saves, and follower growth.
- Recognize what makes a tweet timely, shareable, and discussion-worthy.
- Adjust style and structure based on what tends to perform better with the intended audience.
- Suggest variations or alternate angles when multiple posting approaches could improve performance.

For every draft you receive, your workflow should include:
- Understanding the source content
- Identifying the core message
- Selecting the most effective social angle
- Writing a concise and engaging tweet
- Optimizing it for platform best practices
- Checking clarity, compliance, and brand consistency
- Recommending a thread if the content needs more space

When creating output, provide:
- A final tweet version
- 2 to 3 alternative tweet options when useful
- Suggested hashtags if relevant
- Suggested mentions if relevant
- A thread version if the topic cannot be covered effectively in one tweet
- A brief note on the strategic angle or engagement goal when needed

Your working style should be:
- Concise
- Sharp
- Audience-aware
- Brand-consistent
- Ethical
- Engagement-focused

Your goal is not just to shorten content, but to turn research and draft material into polished Twitter content that captures attention, communicates clearly, respects platform norms, and strengthens the brand’s presence online."""
)

social_media_manager_node = functools.partial(
    agent_node, agent=social_media_manager_agent, name="social_media_manager"
)

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    next: str

workflow = StateGraph(AgentState)
workflow.add_node("content_marketing_manager", content_marketing_manager_chain)
workflow.add_node("online_researcher", online_researcher_node)
workflow.add_node("blog_manager", blog_manager_node)
workflow.add_node("social_media_manager", social_media_manager_node)

for member in content_marketing_team:
    workflow.add_edge(member, "content_marketing_manager")

conditional_map = {k: k for k in content_marketing_team}
conditional_map["FINISH"] = END

workflow.add_conditional_edges(
    "content_marketing_manager", lambda x: x["next"], conditional_map
)

workflow.set_entry_point("content_marketing_manager")
multiagent = workflow.compile()

print("Iniciando flujo...")
with get_openai_callback() as cb:
    events = multiagent.stream(
        {
            "messages": [
                HumanMessage(
                    content="Write me a report on the power of social media. After the research on the power of social media, pass the findings to the blog manager to generate the final blog article. Once done, pass it to the social media manager to write a tweet on the subject."
                )
            ]
        },
        {"recursion_limit": 150},
    )

    for event in events:
        print(event)
        print("\n-----------------\n")
        if cb.total_cost >= MAX_BUDGET_USD:
            print(
                f"Presupuesto alcanzado: ${cb.total_cost:.4f} >= ${MAX_BUDGET_USD:.4f}. Deteniendo flujo."
            )
            break

    print(
        f"Costo total: ${cb.total_cost:.4f} | prompt_tokens={cb.prompt_tokens} | completion_tokens={cb.completion_tokens} | total_tokens={cb.total_tokens}"
    )

print("Flujo terminado.")
