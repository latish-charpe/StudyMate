"""
StudyMate - Functional Tool-Using AI Study Agent Orchestration Module
Provides explicit, understandable tool routing, multi-step task execution,
result-grounded decision making, and student-friendly feedback.
"""

from __future__ import annotations
import json
import re
import logging
from typing import Any, Callable, Dict, List, Optional, Tuple
from dataclasses import dataclass, field, asdict

# Configure quiet logger for developer debugging
logger = logging.getLogger("studymate_agent")
logger.setLevel(logging.INFO)


# ============================================================================
# DATA STRUCTURES FOR AGENT PLANS AND STEPS
# ============================================================================

@dataclass
class AgentStep:
    id: int
    title: str
    tool_name: str
    tool_input: Dict[str, Any]
    status: str = "pending"  # "pending", "running", "completed", "waiting_user_input", "requires_approval", "failed"
    status_message: str = "Pending"
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    requires_approval: bool = False
    approved: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class AgentExecutionState:
    goal: str
    steps: List[AgentStep] = field(default_factory=list)
    current_step_idx: int = 0
    is_finished: bool = False
    context_memory: Dict[str, Any] = field(default_factory=dict)
    summary_message: Optional[str] = None
    error_message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "goal": self.goal,
            "steps": [s.to_dict() for s in self.steps],
            "current_step_idx": self.current_step_idx,
            "is_finished": self.is_finished,
            "context_memory": self.context_memory,
            "summary_message": self.summary_message,
            "error_message": self.error_message,
        }


# ============================================================================
# TOOL IMPLEMENTATIONS
# ============================================================================

class StudyAgentTools:
    """
    Core functional study tools implemented with strict accuracy and validation.
    """

    def __init__(self, llm_invoker: Callable[[str], str], vectorstore_getter: Callable[[], Any]):
        self.ask_llm = llm_invoker
        self.get_vectorstore = vectorstore_getter

    # -------------------------------------------------------------------------
    # Tool 1: Search Study Notes
    # -------------------------------------------------------------------------
    def search_study_notes(self, query: str, k: int = 4) -> Dict[str, Any]:
        """
        Search the processed PDF index, retrieve relevant passages,
        preserve filenames and page numbers, and keep grounded in retrieved content.
        """
        query_str = str(query).strip() if query else "key concepts and main definitions"
        db = self.get_vectorstore()

        if db is None:
            return {
                "success": False,
                "error": "No study notes have been uploaded and indexed yet. Please upload your course PDF first.",
                "passages": [],
                "sources": [],
                "formatted_context": "",
                "direct_answer": "No indexed study material is available to search.",
            }

        try:
            docs = db.similarity_search(query_str, k=k)
            if not docs:
                return {
                    "success": True,
                    "passages": [],
                    "sources": [],
                    "formatted_context": "",
                    "direct_answer": f"No relevant sections matching '{query_str}' were found in your notes.",
                }

            passages = []
            sources = []
            seen_sources = set()
            context_blocks = []

            for i, doc in enumerate(docs, start=1):
                src = doc.metadata.get("source", "Study Document")
                page = doc.metadata.get("page", "?")
                text = doc.page_content.strip()
                passages.append({
                    "source": src,
                    "page": page,
                    "content": text,
                    "snippet": " ".join(text[:200].split()),
                })
                key = (src, page)
                if key not in seen_sources:
                    seen_sources.add(key)
                    sources.append({"source": src, "page": page})

                context_blocks.append(f"[Source {i}: {src}, Page {page}]\n{text}")

            formatted_context = "\n\n".join(context_blocks)

            # Generate grounded answer strictly from retrieved notes
            prompt = f"""You are StudyMate's dedicated study assistant tool.
Answer the query using ONLY the verified excerpts from the student's study notes below.

GROUNDING RULES:
1. Base your statements strictly on the provided study excerpts.
2. Explicitly cite the page numbers and sources (e.g., [Week 3.pdf, Page 2]).
3. If the excerpts do not contain the complete answer, state clearly what is known and what is missing.
4. Structure your response with clear bullet points.

STUDY EXCERPTS:
{formatted_context}

QUERY:
{query_str}

Grounded response:"""

            direct_answer = self.ask_llm(prompt)

            return {
                "success": True,
                "query": query_str,
                "count": len(passages),
                "passages": passages,
                "sources": sources,
                "formatted_context": formatted_context,
                "direct_answer": direct_answer,
            }
        except Exception as e:
            logger.error(f"search_study_notes error: {e}")
            return {
                "success": False,
                "error": f"Error searching study notes: {str(e)}",
                "passages": [],
                "sources": [],
                "formatted_context": "",
                "direct_answer": "An error occurred while querying your study notes.",
            }

    # -------------------------------------------------------------------------
    # Tool 2: Generate Study Summary
    # -------------------------------------------------------------------------
    def generate_study_summary(
        self,
        topic: str = "",
        scope: str = "targeted",
        pre_retrieved_context: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Create concise summaries from the uploaded study material.
        Preserves important definitions, concepts, formulas, and examples.
        Reuses retrieved content where appropriate to avoid unnecessary API calls.
        """
        db = self.get_vectorstore()
        context_to_use = pre_retrieved_context or ""

        if not context_to_use:
            if db is None:
                return {
                    "success": False,
                    "error": "No study notes available to summarize. Please upload your course PDF first.",
                    "summary": "",
                    "topic": topic,
                }

            if scope == "entire" or not topic.strip():
                # Summarize entire doc collection safely up to character budget
                all_chunks = list(db.docstore._dict.values())
                if not all_chunks:
                    return {
                        "success": False,
                        "error": "The document index has no readable passages.",
                        "summary": "",
                        "topic": topic or "All Material",
                    }
                unique_texts = []
                seen = set()
                total_len = 0
                for c in all_chunks:
                    snippet = c.page_content.strip()
                    key = snippet[:100]
                    if key not in seen:
                        seen.add(key)
                        src = c.metadata.get("source", "Notes")
                        p = c.metadata.get("page", "?")
                        block = f"[{src}, Page {p}]: {snippet}"
                        if total_len + len(block) < 30000:
                            unique_texts.append(block)
                            total_len += len(block)
                        else:
                            break
                context_to_use = "\n\n".join(unique_texts)
                topic_label = "Complete Study Material"
            else:
                # Targeted topic summary
                search_res = self.search_study_notes(query=topic, k=6)
                if not search_res.get("success") or not search_res.get("passages"):
                    return {
                        "success": False,
                        "error": f"No notes found regarding '{topic}' to summarize.",
                        "summary": "",
                        "topic": topic,
                    }
                context_to_use = search_res.get("formatted_context", "")
                topic_label = topic
        else:
            topic_label = topic if topic.strip() else "Selected Study Material"

        prompt = f"""You are StudyMate's educational summarizer tool.
Produce a high-yield, structured revision summary based STRICTLY on the study notes below.

REQUIRED STRUCTURE:
1. **Executive Overview**: 2-3 sentence core synthesis.
2. **Key Definitions & Principles**: Essential terminology and foundational concepts.
3. **Core Frameworks & Comparative Breakdown**: Detailed analysis, comparisons, or formulas.
4. **Important Examples / Applications**: Concrete illustrations mentioned in the notes.
5. **High-Yield Exam Takeaways**: Bullet points of critical facts for quick review.

GROUNDING RULES:
- Preserve technical definitions, formulas, and theories accurately.
- Do not invent facts outside of the provided material.
- Use clean Markdown with headers, bold terms, and bullet points.

STUDY MATERIAL:
{context_to_use}

TOPIC FOCUS: {topic_label}

Structured Revision Summary:"""

        try:
            summary = self.ask_llm(prompt)
            word_count = len(summary.split())
            return {
                "success": True,
                "topic": topic_label,
                "summary": summary,
                "word_count": word_count,
                "estimated_read_min": max(1, round(word_count / 200)),
            }
        except Exception as e:
            logger.error(f"generate_study_summary error: {e}")
            return {
                "success": False,
                "error": f"Error generating study summary: {str(e)}",
                "summary": "",
                "topic": topic_label,
            }

    # -------------------------------------------------------------------------
    # Tool 3: Generate and Evaluate Quiz
    # -------------------------------------------------------------------------
    def generate_quiz(
        self,
        topic: str = "",
        count: int = 5,
        pre_retrieved_context: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate multiple-choice questions grounded in uploaded notes.
        Returns parsed questions with distinct options, correct answers, explanations, and topic tags.
        """
        db = self.get_vectorstore()
        context_to_use = pre_retrieved_context or ""

        if not context_to_use:
            if db is None:
                return {
                    "success": False,
                    "error": "No study notes uploaded. Please upload course PDFs before generating a quiz.",
                    "questions": [],
                }
            query = topic.strip() if topic.strip() else "key concepts, definitions, principles, and theories"
            search_res = self.search_study_notes(query=query, k=8)
            if not search_res.get("success") or not search_res.get("passages"):
                return {
                    "success": False,
                    "error": f"Could not find sufficient study material to generate a quiz on '{topic}'.",
                    "questions": [],
                }
            context_to_use = search_res.get("formatted_context", "")

        count = max(1, min(count, 10))

        prompt = f"""You are an academic assessment generation engine.
Create exactly {count} high-quality multiple-choice questions based STRICTLY on the study material below.

RULES:
1. Every question must test a factual concept explicitly described in the notes.
2. Provide exactly 4 options labeled A, B, C, D.
3. Exactly one option is correct.
4. Include a concise educational explanation citing why the answer is correct based on the notes.
5. Include a specific concept/topic tag for each question (e.g. "Management Roles", "ETL Process", "Supervised Learning").
6. Output MUST BE strictly a valid JSON object matching the schema below. No markdown backticks, no prose.

JSON SCHEMA:
{{
  "questions": [
    {{
      "id": 1,
      "topic": "Topic/Concept Name",
      "question": "Question text here?",
      "options": {{
        "A": "Option text 1",
        "B": "Option text 2",
        "C": "Option text 3",
        "D": "Option text 4"
      }},
      "answer": "A",
      "explanation": "Explanation grounded in the text."
    }}
  ]
}}

STUDY MATERIAL:
{context_to_use}

TOPIC FOCUS: {topic or 'Core Course Material'}

JSON:"""

        try:
            raw = self.ask_llm(prompt)
            cleaned = raw.strip()
            # Strip markdown code blocks if returned
            if cleaned.startswith("```"):
                lines = cleaned.splitlines()
                if lines and lines[0].startswith("```"):
                    lines = lines[1:]
                if lines and lines[-1].strip() == "```":
                    lines = lines[:-1]
                cleaned = "\n".join(lines).strip()
            if cleaned.lower().startswith("json"):
                cleaned = cleaned[4:].strip()

            start_idx = cleaned.find("{")
            end_idx = cleaned.rfind("}")
            if start_idx != -1 and end_idx != -1:
                cleaned = cleaned[start_idx : end_idx + 1]

            parsed = json.loads(cleaned)
            raw_qs = parsed.get("questions", [])

            normalized = []
            for idx, item in enumerate(raw_qs, start=1):
                q_text = str(item.get("question", "")).strip()
                opts = item.get("options", {})
                ans = str(item.get("answer", "")).strip().upper()
                expl = str(item.get("explanation", "")).strip()
                topic_tag = str(item.get("topic", topic or "General Concepts")).strip()

                if isinstance(opts, list):
                    labels = ["A", "B", "C", "D"]
                    opts = {labels[i]: str(v).strip() for i, v in enumerate(opts[:4])}
                elif isinstance(opts, dict):
                    opts = {str(k).strip().upper(): str(v).strip() for k, v in opts.items()}

                match = re.match(r"^([A-D])", ans)
                ans_letter = match.group(1) if match else ans[:1]

                if q_text and len(opts) >= 2 and ans_letter in opts:
                    normalized.append({
                        "id": idx,
                        "topic": topic_tag,
                        "question": q_text,
                        "options": opts,
                        "answer": ans_letter,
                        "explanation": expl,
                    })

            if not normalized:
                return {
                    "success": False,
                    "error": "Failed to extract valid multiple choice questions from notes.",
                    "questions": [],
                }

            return {
                "success": True,
                "count": len(normalized),
                "topic": topic or "General Concepts",
                "questions": normalized,
            }

        except Exception as e:
            logger.error(f"generate_quiz error: {e}")
            return {
                "success": False,
                "error": f"Error generating quiz questions: {str(e)}",
                "questions": [],
            }

    def evaluate_quiz(self, questions: List[Dict[str, Any]], user_answers: Dict[int, str]) -> Dict[str, Any]:
        """
        Check submitted answers, calculate score, explain correct answers,
        and identify topics associated with incorrect answers.
        """
        if not questions:
            return {
                "success": False,
                "error": "No quiz questions provided for evaluation.",
                "score": 0,
                "total": 0,
                "percentage": 0,
                "breakdown": [],
                "weak_topics": [],
            }

        total = len(questions)
        correct_count = 0
        breakdown = []
        weak_topics_set = set()
        topic_performance: Dict[str, Dict[str, int]] = {}

        for i, q in enumerate(questions):
            ans_correct = str(q.get("answer", "")).strip().upper()
            selected = str(user_answers.get(i, "")).strip().upper()
            opts = q.get("options", {})
            topic_tag = q.get("topic", "General")

            if topic_tag not in topic_performance:
                topic_performance[topic_tag] = {"total": 0, "correct": 0}
            topic_performance[topic_tag]["total"] += 1

            is_correct = bool(selected and selected == ans_correct)
            if is_correct:
                correct_count += 1
                topic_performance[topic_tag]["correct"] += 1
            else:
                weak_topics_set.add(topic_tag)

            breakdown.append({
                "index": i + 1,
                "topic": topic_tag,
                "question": q.get("question", ""),
                "selected_option": selected if selected else "Unanswered",
                "selected_text": opts.get(selected, "Not answered"),
                "correct_option": ans_correct,
                "correct_text": opts.get(ans_correct, ""),
                "is_correct": is_correct,
                "explanation": q.get("explanation", ""),
            })

        percentage = round((correct_count / total) * 100) if total > 0 else 0

        # Formulate list of weak topics with accuracy info
        weak_topics = []
        for t, stats in topic_performance.items():
            t_acc = round((stats["correct"] / stats["total"]) * 100) if stats["total"] > 0 else 0
            if t_acc < 100 or stats["correct"] < stats["total"]:
                weak_topics.append({
                    "topic": t,
                    "correct": stats["correct"],
                    "total": stats["total"],
                    "accuracy": t_acc,
                })

        # Performance tier
        if percentage >= 85:
            tier = "Mastery"
            feedback = "Excellent grasp of the core concepts! You demonstrated strong retention."
        elif percentage >= 60:
            tier = "Competent"
            feedback = "Solid foundation with a few specific conceptual gaps to address."
        else:
            tier = "Needs Revision"
            feedback = "Targeted revision needed on identified weak topics before attempting new material."

        return {
            "success": True,
            "score": correct_count,
            "total": total,
            "percentage": percentage,
            "tier": tier,
            "feedback": feedback,
            "breakdown": breakdown,
            "weak_topics": [w["topic"] for w in weak_topics],
            "weak_topics_details": weak_topics,
        }

    # -------------------------------------------------------------------------
    # Tool 4: Create Study Plan
    # -------------------------------------------------------------------------
    def create_study_plan(
        self,
        subject: str,
        topics: Optional[List[str]] = None,
        study_hours_per_day: float = 2.0,
        exam_timeframe: str = "7 days",
        available_notes_topics: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Collect student's subject, topics, available study time, and exam timeframe.
        Prioritizes topics using available notes and student's stated goals.
        Creates a structured daily revision plan clearly distinguishing suggested plans
        from actions actually completed.
        """
        # Determine actual available topics from notes if not supplied
        db = self.get_vectorstore()
        doc_sources = []
        if db is not None:
            try:
                for doc in db.docstore._dict.values():
                    s = doc.metadata.get("source")
                    if s and s not in doc_sources:
                        doc_sources.append(s)
            except Exception:
                pass

        topics_list = topics or []
        if not topics_list and available_notes_topics:
            topics_list = available_notes_topics

        topics_str = ", ".join(topics_list) if topics_list else "All core modules from indexed notes"
        sources_str = ", ".join(doc_sources) if doc_sources else "Uploaded study material"

        prompt = f"""You are StudyMate's educational planner tool.
Create a realistic, structured, daily study revision plan for a student.

STUDENT STUDY PROFILE:
- Subject/Course: {subject or 'College Course'}
- Topics to Cover: {topics_str}
- Daily Study Time Available: {study_hours_per_day} hours/day
- Target Exam Timeframe / Deadline: {exam_timeframe}
- Available Course Materials Indexed: {sources_str}

REQUIREMENTS:
1. Divide the preparation into logical daily blocks (e.g. Day 1, Day 2...).
2. For each day, specify:
   - Priority Topic
   - Dedicated Study Time (hours)
   - Specific Learning Tasks (Reading notes, Active recall, Practice problems)
   - Daily Checkpoint / Milestone
3. Prioritize higher-weight foundational topics first, followed by application and final revision.
4. Explicitly mark all upcoming days as "Scheduled / Suggested Plan" (DO NOT claim them as completed).
5. State realistic revision strategies (spaced repetition, Feynman technique).

Format with clean Markdown, bold headers, and structured daily tables or bullet blocks."""

        try:
            plan_text = self.ask_llm(prompt)
            return {
                "success": True,
                "subject": subject,
                "topics": topics_list,
                "daily_hours": study_hours_per_day,
                "exam_timeframe": exam_timeframe,
                "sources_used": doc_sources,
                "plan_markdown": plan_text,
                "status": "Plan Generated (Awaiting Student Execution)",
            }
        except Exception as e:
            logger.error(f"create_study_plan error: {e}")
            return {
                "success": False,
                "error": f"Error creating study plan: {str(e)}",
                "plan_markdown": "",
            }

    # -------------------------------------------------------------------------
    # Tool 5: Recommend Next Action
    # -------------------------------------------------------------------------
    def recommend_next_action(
        self,
        quiz_evaluation: Optional[Dict[str, Any]] = None,
        student_goal: str = "",
        weak_topics: Optional[List[str]] = None,
        completed_steps: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Use quiz results, student goals, and available document content to recommend what to study next.
        Bases recommendations on actual results, not fabricated progress.
        """
        score_info = ""
        weak_str = ""

        if quiz_evaluation and quiz_evaluation.get("success"):
            score = quiz_evaluation.get("score", 0)
            total = quiz_evaluation.get("total", 0)
            pct = quiz_evaluation.get("percentage", 0)
            score_info = f"Quiz Score: {score}/{total} ({pct}%)"
            w_list = quiz_evaluation.get("weak_topics", [])
            if w_list:
                weak_str = f"Identified Weak Topics: {', '.join(w_list)}"
            else:
                weak_str = "No weak topics detected (100% correct answers)"
        elif weak_topics:
            weak_str = f"Target Topics: {', '.join(weak_topics)}"
        else:
            weak_str = "No prior quiz results yet. Diagnostic assessment recommended."

        completed_str = ", ".join(completed_steps) if completed_steps else "Goal initiated"

        prompt = f"""You are StudyMate's adaptive educational guidance advisor.
Based strictly on the student's actual performance data and goals below, recommend the next concrete study action.

STUDENT DATA:
- Stated Learning Goal: {student_goal or 'Prepare for exam'}
- Current Performance: {score_info or 'Assessment in progress'}
- {weak_str}
- Completed Steps So Far: {completed_str}

RULES:
1. Base your recommendation ONLY on actual data. Do not fabricate hypothetical quiz scores.
2. If the student has weak topics (<70% or missed questions), prioritize a targeted 15-minute review of those exact concepts, followed by a short focused re-quiz.
3. If the student scored high (>85%), suggest moving forward to advanced applications or next scheduled syllabus topics.
4. Provide:
   - Primary Recommended Action (single actionable sentence)
   - Why This Next Step (pedagogical rationale)
   - Suggested Method (e.g. read notes page X, make flashcards, re-take 3 MCQs)
   - Follow-up Checkpoint

Keep it concise, encouraging, and actionable."""

        try:
            recommendation_text = self.ask_llm(prompt)
            # Determine actionable tool recommendation
            suggested_tool = "search_study_notes"
            suggested_param = ""
            if quiz_evaluation and quiz_evaluation.get("weak_topics"):
                suggested_tool = "search_study_notes"
                suggested_param = quiz_evaluation["weak_topics"][0]
            elif quiz_evaluation and quiz_evaluation.get("percentage", 0) >= 80:
                suggested_tool = "create_study_plan"
                suggested_param = "next syllabus unit"

            return {
                "success": True,
                "recommendation_markdown": recommendation_text,
                "suggested_next_tool": suggested_tool,
                "suggested_parameter": suggested_param,
                "based_on_weak_topics": quiz_evaluation.get("weak_topics", []) if quiz_evaluation else [],
            }
        except Exception as e:
            logger.error(f"recommend_next_action error: {e}")
            return {
                "success": False,
                "error": f"Error generating recommendation: {str(e)}",
                "recommendation_markdown": "Review your course notes and continue with your study plan.",
                "suggested_next_tool": "search_study_notes",
                "suggested_parameter": "",
            }

    # -------------------------------------------------------------------------
    # Tool 6: General Study Assistant
    # -------------------------------------------------------------------------
    def general_study_assistant(self, question: str, subject_context: str = "") -> Dict[str, Any]:
        """
        Answer general educational questions using LLM knowledge.
        Keeps this strictly separate from document-grounded answers.
        Clearly distinguishes general knowledge from information retrieved from uploaded notes.
        """
        q_clean = question.strip() if question else "Explain this concept"

        prompt = f"""You are StudyMate's general academic AI tutor.
The student is asking a general subject question not grounded in a specific uploaded document.

Provide a clear, engaging, step-by-step academic explanation.
Include:
- Clear definition and intuition
- Core principles / formula / code if relevant
- Common pitfalls or misconceptions
- A quick memory hook or mnemonic

CONTEXT/SUBJECT: {subject_context or 'General Academics'}
STUDENT QUESTION: {q_clean}

Comprehensive Tutorial Answer:"""

        try:
            answer = self.ask_llm(prompt)
            return {
                "success": True,
                "question": q_clean,
                "is_general_knowledge": True,
                "disclaimer": "💡 General Academic Knowledge (Not derived from uploaded notes)",
                "answer": answer,
            }
        except Exception as e:
            logger.error(f"general_study_assistant error: {e}")
            return {
                "success": False,
                "error": f"Error answering question: {str(e)}",
                "is_general_knowledge": True,
                "answer": "Could not complete general tutoring request at this time.",
            }


# ============================================================================
# AGENT CONTROLLER / ORCHESTRATION LAYER
# ============================================================================

class StudyAgentController:
    """
    Orchestrates student study goals by:
    - Understanding user requests and breaking multi-step goals into concrete tasks.
    - Deciding appropriate tools and validating prerequisites.
    - Executing tools sequentially with student-friendly activity feedback.
    - Handling human-in-the-loop interactions (e.g. taking the quiz, approving plans).
    - Re-evaluating next actions based on real results.
    - Halting safely when limits are hit or resources are missing.
    """

    ACTIVITY_MESSAGES = {
        "search_study_notes": "Searching your notes",
        "generate_study_summary": "Preparing your revision summary",
        "generate_quiz": "Generating practice questions",
        "evaluate_quiz": "Evaluating your answers",
        "create_study_plan": "Preparing your study plan",
        "recommend_next_action": "Recommending your next revision task",
        "general_study_assistant": "Consulting academic knowledge base",
    }

    def __init__(self, tools: StudyAgentTools):
        self.tools = tools
        self.max_execution_steps = 8

    def analyze_goal_and_plan(self, goal: str, available_sources: List[str]) -> AgentExecutionState:
        """
        Decomposes the student's request into a transparent, actionable sequence of steps.
        Explicitly recognizes both multi-step workflows and direct single-tool requests.
        """
        clean_goal = goal.strip()
        lower_goal = clean_goal.lower()
        steps: List[AgentStep] = []

        has_docs = bool(available_sources)

        # Detect the classic comprehensive exam preparation workflow
        # Example: "Prepare me for my exam using my uploaded notes. Create a study plan, test my knowledge, and tell me what I should revise next."
        is_full_prep = (
            ("prepare" in lower_goal or "exam" in lower_goal or "study plan" in lower_goal)
            and ("test" in lower_goal or "quiz" in lower_goal or "practice" in lower_goal)
            and ("revise" in lower_goal or "next" in lower_goal or "recommend" in lower_goal)
        )

        if is_full_prep:
            # 1. Search/Inspect notes to understand indexed syllabus
            steps.append(
                AgentStep(
                    id=1,
                    title="Inspect Study Material",
                    tool_name="search_study_notes",
                    tool_input={"query": "syllabus, core concepts, main chapters, and principles", "k": 6},
                    status_message="Searching your notes for key concepts",
                )
            )
            # 2. Create structured study plan
            steps.append(
                AgentStep(
                    id=2,
                    title="Generate Structured Study Plan",
                    tool_name="create_study_plan",
                    tool_input={
                        "subject": "Course Exam Preparation",
                        "study_hours_per_day": 2.0,
                        "exam_timeframe": "7-day intensive revision",
                    },
                    status_message="Preparing your study plan",
                    requires_approval=True,  # Student approves study plan schedule
                )
            )
            # 3. Generate diagnostic quiz
            steps.append(
                AgentStep(
                    id=3,
                    title="Generate Diagnostic Practice Quiz",
                    tool_name="generate_quiz",
                    tool_input={"topic": "core concepts and definitions", "count": 5},
                    status_message="Generating practice questions from your notes",
                )
            )
            # 4. Interactive Quiz Evaluation step (Waits for student to answer and submit!)
            steps.append(
                AgentStep(
                    id=4,
                    title="Evaluate Quiz & Identify Weak Topics",
                    tool_name="evaluate_quiz",
                    tool_input={},  # Fed dynamically when student submits answers
                    status="waiting_user_input",
                    status_message="Awaiting your quiz submission",
                )
            )
            # 5. Recommendation based on real quiz outcome
            steps.append(
                AgentStep(
                    id=5,
                    title="Recommend Next Targeted Revision Task",
                    tool_name="recommend_next_action",
                    tool_input={},
                    status="pending",
                    status_message="Recommending your next revision task",
                )
            )

        elif "quiz" in lower_goal or "test" in lower_goal or "mcq" in lower_goal:
            # Quiz creation workflow
            topic = self._extract_topic_keyword(clean_goal, ["quiz", "test", "on", "about", "for", "mcq"])
            count_match = re.search(r"(\d+)\s*(?:questions?|mcqs?|items?)", lower_goal)
            q_count = int(count_match.group(1)) if count_match else 5
            q_count = min(max(q_count, 3), 10)

            steps.append(
                AgentStep(
                    id=1,
                    title=f"Generate Quiz on {topic or 'Notes'}",
                    tool_name="generate_quiz",
                    tool_input={"topic": topic, "count": q_count},
                    status_message="Generating practice questions",
                )
            )
            steps.append(
                AgentStep(
                    id=2,
                    title="Evaluate Answers & Weak Topics",
                    tool_name="evaluate_quiz",
                    tool_input={},
                    status="waiting_user_input",
                    status_message="Awaiting your quiz submission",
                )
            )
            steps.append(
                AgentStep(
                    id=3,
                    title="Recommend Next Revision Focus",
                    tool_name="recommend_next_action",
                    tool_input={},
                    status="pending",
                    status_message="Recommending your next revision task",
                )
            )

        elif "summary" in lower_goal or "summarize" in lower_goal or "overview" in lower_goal:
            topic = self._extract_topic_keyword(clean_goal, ["summarize", "summary", "of", "about", "notes on"])
            steps.append(
                AgentStep(
                    id=1,
                    title=f"Generate Revision Summary{' for ' + topic if topic else ''}",
                    tool_name="generate_study_summary",
                    tool_input={"topic": topic, "scope": "targeted" if topic else "entire"},
                    status_message="Preparing your revision summary",
                )
            )

        elif "study plan" in lower_goal or "plan" in lower_goal or "schedule" in lower_goal:
            subject = self._extract_topic_keyword(clean_goal, ["plan", "for", "study", "schedule", "on"]) or "Course Revision"
            steps.append(
                AgentStep(
                    id=1,
                    title="Inspect Available Topics in Notes",
                    tool_name="search_study_notes",
                    tool_input={"query": "table of contents, syllabus topics, chapters", "k": 5},
                    status_message="Searching your notes for syllabus outline",
                )
            )
            steps.append(
                AgentStep(
                    id=2,
                    title=f"Create Custom Study Plan: {subject}",
                    tool_name="create_study_plan",
                    tool_input={
                        "subject": subject,
                        "study_hours_per_day": 2.0,
                        "exam_timeframe": "7 days",
                    },
                    status_message="Preparing your study plan",
                )
            )

        elif any(k in lower_goal for k in ["find", "search", "where", "what does", "explain from notes", "page"]):
            # Document Search
            steps.append(
                AgentStep(
                    id=1,
                    title="Search Notes & Retrieve Grounded Content",
                    tool_name="search_study_notes",
                    tool_input={"query": clean_goal, "k": 4},
                    status_message="Searching your notes",
                )
            )

        else:
            # If documents exist, check notes first, then provide educational guidance
            if has_docs:
                steps.append(
                    AgentStep(
                        id=1,
                        title="Search Grounded Material in Notes",
                        tool_name="search_study_notes",
                        tool_input={"query": clean_goal, "k": 4},
                        status_message="Searching your notes",
                    )
                )
            else:
                steps.append(
                    AgentStep(
                        id=1,
                        title="General Academic Tutoring",
                        tool_name="general_study_assistant",
                        tool_input={"question": clean_goal},
                        status_message="Consulting academic knowledge base",
                    )
                )

        state = AgentExecutionState(
            goal=clean_goal,
            steps=steps,
            current_step_idx=0,
            is_finished=False,
            context_memory={"available_sources": available_sources},
        )
        return state

    def execute_next_step(
        self,
        state: AgentExecutionState,
        user_interaction_payload: Optional[Dict[str, Any]] = None,
    ) -> AgentExecutionState:
        """
        Executes one step in the agent plan, updates state, inspects results,
        and dynamically determines whether to pause for user input, request approval, or proceed.
        """
        if state.is_finished:
            return state

        if state.current_step_idx >= len(state.steps):
            state.is_finished = True
            state.summary_message = "All agent tasks for this study goal have been successfully concluded."
            return state

        current_step = state.steps[state.current_step_idx]

        # Check if step requires approval and has not been approved yet
        if current_step.requires_approval and not current_step.approved:
            current_step.status = "requires_approval"
            current_step.status_message = "Awaiting your confirmation to proceed"
            return state

        # Check if step is waiting for user input (e.g., student answers to the quiz)
        if current_step.status == "waiting_user_input":
            if not user_interaction_payload or not user_interaction_payload.get("answers"):
                # Still waiting for student to submit
                current_step.status_message = "Please complete and submit the quiz above"
                return state
            else:
                # Payload received! Bind answers to step tool_input
                current_step.tool_input["user_answers"] = user_interaction_payload["answers"]
                current_step.status = "pending"

        # Execute the tool
        current_step.status = "running"
        current_step.status_message = self.ACTIVITY_MESSAGES.get(current_step.tool_name, "Processing task")

        try:
            result = self._dispatch_tool(current_step.tool_name, current_step.tool_input, state)
            current_step.result = result

            if result.get("success", False):
                current_step.status = "completed"
                current_step.status_message = "Completed"

                # Save useful outputs into shared context memory for subsequent tools
                self._update_context_memory(state, current_step)

                # Move to next step index
                state.current_step_idx += 1

                # If this was the last step, mark finished
                if state.current_step_idx >= len(state.steps):
                    state.is_finished = True
                    self._generate_final_summary(state)

            else:
                # Tool reported a controlled error/failure
                current_step.status = "failed"
                current_step.error = result.get("error", "The tool could not complete this task.")
                current_step.status_message = f"Failed: {current_step.error}"
                state.error_message = current_step.error
                # Stop safely
                state.is_finished = True

        except Exception as e:
            logger.error(f"Execution error on step {current_step.id}: {e}")
            current_step.status = "failed"
            current_step.error = str(e)
            current_step.status_message = "Encountered an error while running task"
            state.error_message = str(e)
            state.is_finished = True

        return state

    def _dispatch_tool(self, tool_name: str, tool_input: Dict[str, Any], state: AgentExecutionState) -> Dict[str, Any]:
        """Routes execution to the matching tool function with input validation."""
        if tool_name == "search_study_notes":
            query = tool_input.get("query", state.goal)
            k = tool_input.get("k", 4)
            return self.tools.search_study_notes(query=query, k=k)

        elif tool_name == "generate_study_summary":
            topic = tool_input.get("topic", "")
            scope = tool_input.get("scope", "targeted")
            pre_context = state.context_memory.get("last_search_context")
            return self.tools.generate_study_summary(topic=topic, scope=scope, pre_retrieved_context=pre_context)

        elif tool_name == "generate_quiz":
            topic = tool_input.get("topic", "")
            count = tool_input.get("count", 5)
            pre_context = state.context_memory.get("last_search_context")
            return self.tools.generate_quiz(topic=topic, count=count, pre_retrieved_context=pre_context)

        elif tool_name == "evaluate_quiz":
            # Questions come from prior generate_quiz step
            questions = state.context_memory.get("quiz_questions", [])
            user_answers = tool_input.get("user_answers", {})
            return self.tools.evaluate_quiz(questions=questions, user_answers=user_answers)

        elif tool_name == "create_study_plan":
            subject = tool_input.get("subject", "Course Revision")
            topics = tool_input.get("topics")
            hrs = tool_input.get("study_hours_per_day", 2.0)
            timeframe = tool_input.get("exam_timeframe", "7 days")
            avail_topics = state.context_memory.get("identified_topics")
            return self.tools.create_study_plan(
                subject=subject,
                topics=topics,
                study_hours_per_day=hrs,
                exam_timeframe=timeframe,
                available_notes_topics=avail_topics,
            )

        elif tool_name == "recommend_next_action":
            quiz_eval = state.context_memory.get("last_quiz_evaluation")
            weak_topics = state.context_memory.get("weak_topics")
            completed_steps_names = [s.title for s in state.steps if s.status == "completed"]
            return self.tools.recommend_next_action(
                quiz_evaluation=quiz_eval,
                student_goal=state.goal,
                weak_topics=weak_topics,
                completed_steps=completed_steps_names,
            )

        elif tool_name == "general_study_assistant":
            q = tool_input.get("question", state.goal)
            return self.tools.general_study_assistant(question=q)

        else:
            return {"success": False, "error": f"Unknown tool: '{tool_name}'"}

    def _update_context_memory(self, state: AgentExecutionState, step: AgentStep):
        """Passes verified outputs between tool steps to maintain state coherency."""
        res = step.result or {}

        if step.tool_name == "search_study_notes":
            state.context_memory["last_search_context"] = res.get("formatted_context", "")
            state.context_memory["last_search_sources"] = res.get("sources", [])

        elif step.tool_name == "generate_quiz":
            state.context_memory["quiz_questions"] = res.get("questions", [])

        elif step.tool_name == "evaluate_quiz":
            state.context_memory["last_quiz_evaluation"] = res
            state.context_memory["weak_topics"] = res.get("weak_topics", [])

        elif step.tool_name == "create_study_plan":
            state.context_memory["study_plan_markdown"] = res.get("plan_markdown", "")

        elif step.tool_name == "recommend_next_action":
            state.context_memory["recommendation"] = res.get("recommendation_markdown", "")

    def _generate_final_summary(self, state: AgentExecutionState):
        """Builds a factual, transparent wrap-up distinguishing completed actions from future work."""
        completed = [s for s in state.steps if s.status == "completed"]
        completed_titles = [f"✅ {s.title}" for s in completed]

        eval_data = state.context_memory.get("last_quiz_evaluation")
        quiz_summary = ""
        if eval_data and eval_data.get("success"):
            score = eval_data.get("score")
            total = eval_data.get("total")
            pct = eval_data.get("percentage")
            weaks = eval_data.get("weak_topics", [])
            weak_txt = f" Weak areas identified: **{', '.join(weaks)}**." if weaks else " Outstanding: No conceptual gaps detected."
            quiz_summary = f"\n- **Diagnostic Assessment Result**: {score}/{total} ({pct}%).{weak_txt}"

        state.summary_message = (
            f"**Mission Accomplished**: The agent has completed {len(completed)} orchestrated task(s) for your goal.\n\n"
            f"**Completed Milestones**:\n" + "\n".join(f"- {t}" for t in completed_titles) +
            quiz_summary +
            "\n\n*Note: Future daily revision sessions in your study plan are scheduled for you to carry out and will be tracked as you proceed.*"
        )

    def _extract_topic_keyword(self, text: str, triggers: List[str]) -> str:
        """Helper to extract topic phrase after keywords."""
        pattern = r"(?:" + "|".join(re.escape(t) for t in triggers) + r")\s+([A-Za-z0-9\s_\-\+]+)"
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            candidate = match.group(1).strip()
            # Clean up trailing conjunctions or sentence tails
            candidate = re.split(r"\b(?:and|with|using|in|for|please|now)\b", candidate, flags=re.IGNORECASE)[0].strip()
            return candidate[:60]
        return ""
