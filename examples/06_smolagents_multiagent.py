"""
Модуль 1: Multi-Agent System на SmolAgents
Демонстрація "мультиагентності" через послідовне виконання задач
Логіка агентів: Researcher → Analyst → Reporter
"""

import os
import json
from datetime import datetime
from typing import Dict, List, Any
from smolagents import CodeAgent, tool, ApiModel, OpenAIServerModel
from smolagents.monitoring import LogLevel

# Завантаження .env
try:
    from dotenv import load_dotenv
    load_dotenv()
    print("[OK] .env файл завантажено")
except:
    print("[WARNING] python-dotenv не встановлено")

# ===========================
# ІНСТРУМЕНТИ ДЛЯ АГЕНТІВ
# ===========================

@tool
def search_web(query: str) -> str:
    """
    Пошук інформації в інтернеті через DuckDuckGo.

    Args:
        query: Пошуковий запит

    Returns:
        Результати пошуку
    """
    try:
        from ddgs import DDGS

        results = []
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=3):
                results.append(f"- {r['title']}: {r['body'][:150]}...")

        return f"Результати пошуку для '{query}':\n" + "\n".join(results)
    except Exception as e:
        # Демо результати
        return f"""Демо результати для '{query}':
- AI в освіті: Персоналізація навчання через штучний інтелект
- Тренди 2025: 85% університетів використовують AI
- Виклики: Етика та приватність в AI системах"""

@tool
def analyze_text(text: str) -> str:
    """
    Аналіз тексту та витягування ключової інформації.

    Args:
        text: Текст для аналізу

    Returns:
        Аналітичний висновок
    """
    words = len(text.split())
    sentences = text.count('.') + text.count('!') + text.count('?')

    # Пошук ключових слів
    keywords = {
        'технології': ['AI', 'штучний інтелект', 'machine learning', 'ML'],
        'освіта': ['навчання', 'студенти', 'університет', 'освіта'],
        'тренди': ['тренд', 'майбутнє', '2025', '2024', 'інновація']
    }

    found_keywords = {}
    text_lower = text.lower()

    for category, words_list in keywords.items():
        count = sum(1 for word in words_list if word.lower() in text_lower)
        if count > 0:
            found_keywords[category] = count

    analysis = f"""Аналіз тексту:
- Слів: {words}
- Речень: {sentences}
- Ключові теми: {', '.join(found_keywords.keys()) if found_keywords else 'не виявлено'}
- Деталі: {', '.join([f'{k}({v})' for k, v in found_keywords.items()])}"""

    return analysis

@tool
def save_memory(key: str, value: str) -> str:
    """
    Зберегти дані в локальну пам'ять.

    Args:
        key: Ключ для збереження
        value: Значення для збереження

    Returns:
        Підтвердження збереження
    """
    memory_file = "smolagents_multiagent_memory.json"

    try:
        with open(memory_file, 'r') as f:
            memory = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        memory = {}

    memory[key] = {
        "value": value,
        "timestamp": datetime.now().isoformat()
    }

    with open(memory_file, 'w', encoding='utf-8') as f:
        json.dump(memory, f, ensure_ascii=False, indent=2)

    return f"Збережено: {key}"

@tool
def get_current_time() -> str:
    """
    Отримати поточну дату та час.

    Returns:
        Поточна дата та час у форматі ISO
    """
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# ===========================
# MULTI-AGENT SYSTEM
# ===========================

class SmolAgentsMultiAgentSystem:
    """
    Мультиагентна система на SmolAgents.

    Примітка: SmolAgents не має вбудованої мультиагентності як CrewAI.
    Тут демонструється два підходи:
    1. Один CodeAgent виконує всі три ролі послідовно
    2. Три окремі CodeAgent'и з різними system_prompt'ами
    """

    def __init__(self, model_type: str = "openai", api_key: str = None):
        """Ініціалізація системи"""
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")

        if not self.api_key:
            print("[WARNING] OPENAI_API_KEY не знайдено - працюватиме в обмеженому режимі")
            self.model = None
        else:
            try:
                if model_type == "openai":
                    self.model = OpenAIServerModel(
                        model_id="gpt-4",
                        api_key=self.api_key
                    )
                    print("[OK] OpenAI модель створено")
                elif model_type == "hf":
                    self.model = ApiModel(
                        model_id="meta-llama/Llama-3.3-70B-Instruct",
                        token=os.getenv("HF_TOKEN")
                    )
                    print("[OK] HuggingFace модель створено")
            except Exception as e:
                print(f"[WARNING] Помилка створення моделі: {e}")
                self.model = None

        # Інструменти
        self.tools = [search_web, analyze_text, save_memory, get_current_time]

        # Створення трьох агентів з різними ролями
        self.researcher_agent = self._create_researcher()
        self.analyst_agent = self._create_analyst()
        self.reporter_agent = self._create_reporter()

    def _create_researcher(self):
        """Створення агента-дослідника"""
        if not self.model:
            return None

        instructions_text = """Ви - професійний дослідник з 15-річним стажем.
            Спеціалізуєтесь на освітніх технологіях та штучному інтелекті.
            Ваша задача - знайти найактуальнішу інформацію.

            Використовуйте search_web для пошуку інформації.
            Зберігайте результати через save_memory."""

        return CodeAgent(
            tools=self.tools,
            model=self.model,
            max_steps=2,
            verbosity_level=LogLevel.DEBUG,
            instructions=instructions_text
        )

    def _create_analyst(self):
        """Створення агента-аналітика"""
        if not self.model:
            return None

        instructions_text="""Ви - експерт з data science та аналізу трендів.
            Маєте унікальну здатність знаходити приховані патерни в даних.
            Ваша задача - проаналізувати зібрану інформацію.

            Використовуйте analyze_text для аналізу даних.
            Виявляйте ключові інсайти та тренди."""

        return CodeAgent(
            tools=self.tools,
            model=self.model,
            max_steps=2,
            verbosity_level=LogLevel.DEBUG,
            instructions=instructions_text
        )

    def _create_reporter(self):
        """Створення агента-репортера"""
        if not self.model:
            return None

        instructions_text="""Ви - професійний технічний письменник.
            Вмієте перетворювати складні технічні дані на зрозумілі звіти.
            Ваша задача - створити структурований звіт.

            Створюйте чіткі, зрозумілі звіти для широкої аудиторії."""

        return CodeAgent(
            tools=self.tools,
            model=self.model,
            max_steps=2,
            verbosity_level=LogLevel.DEBUG,
            instructions=instructions_text
        )

    def run_sequential(self, topic: str) -> Dict[str, Any]:
        """
        Підхід 1: Один агент виконує всі задачі послідовно
        """
        print(f"""
╔══════════════════════════════════════════════════════════════╗
║     SMOLAGENTS MULTI-AGENT SYSTEM (Sequential)               ║
║     Один CodeAgent виконує три ролі                          ║
╚══════════════════════════════════════════════════════════════╝

Тема: {topic}
        """)

        if not self.model:
            return self._demo_mode_sequential(topic)

        instructions_text="""Ви - універсальний AI агент з трьома ролями.
            Виконуйте задачу в три етапи:
            1. RESEARCHER: Знайдіть інформацію
            2. ANALYST: Проаналізуйте дані
            3. REPORTER: Створіть звіт"""

        # Створюємо єдиного агента
        agent = CodeAgent(
            tools=self.tools,
            model=self.model,
            max_steps=15,
            verbosity_level=LogLevel.DEBUG,
            instructions=instructions_text
        )

        # Формуємо комплексну задачу
        task = f"""
Дослідіть тему: {topic}

Виконайте три етапи:

ЕТАП 1 - RESEARCH (Дослідження):
1. Використайте search_web для пошуку інформації
2. Збережіть результати через save_memory з ключем "research_results"

ЕТАП 2 - ANALYSIS (Аналіз):
3. Використайте analyze_text для аналізу знайденої інформації
4. Збережіть аналіз через save_memory з ключем "analysis_results"

ЕТАП 3 - REPORTING (Звітність):
5. Створіть структурований звіт з трьох частин:
   - Результати дослідження
   - Аналітичні висновки
   - Рекомендації
6. Збережіть звіт через save_memory з ключем "final_report"

Поверніть фінальний звіт.
        """

        try:
            result = agent.run(task)

            return {
                "approach": "sequential",
                "topic": topic,
                "result": str(result),
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            print(f"[ERROR] Помилка: {e}")
            return {"error": str(e)}

    def run_multi_agent(self, topic: str) -> Dict[str, Any]:
        """
        Підхід 2: Три окремі агенти з різними ролями
        """
        print(f"""
╔══════════════════════════════════════════════════════════════╗
║     SMOLAGENTS MULTI-AGENT SYSTEM (Multi-Agent)              ║
║     Три CodeAgent'и з різними ролями                         ║
╚══════════════════════════════════════════════════════════════╝

Тема: {topic}
Агенти: Researcher → Analyst → Reporter
        """)

        if not self.model:
            return self._demo_mode_multi_agent(topic)

        results = {
            "approach": "multi-agent",
            "topic": topic,
            "timestamp": datetime.now().isoformat()
        }

        # Агент 1: Researcher
        print("\n" + "="*60)
        print("RESEARCHER AGENT: Пошук інформації")
        print("="*60)

        research_task = f"""
Знайдіть інформацію про: {topic}

1. Використайте search_web для пошуку
2. Зберіть мінімум 3 ключові факти
3. Збережіть результати через save_memory з ключем "research_results"

Поверніть знайдену інформацію.
        """

        try:
            research_result = self.researcher_agent.run(research_task)
            results["research"] = str(research_result)
            print(f"[OK] Research завершено")
        except Exception as e:
            print(f"[ERROR] Research помилка: {e}")
            results["research"] = f"Помилка: {e}"
            return results

        # Агент 2: Analyst
        print("\n" + "="*60)
        print("ANALYST AGENT: Аналіз даних")
        print("="*60)

        analysis_task = f"""
Проаналізуйте зібрані дані про: {topic}

Дані для аналізу: {results["research"]}

1. Використайте analyze_text для аналізу
2. Виявіть ключові тренди та інсайти
3. Збережіть аналіз через save_memory з ключем "analysis_results"

Поверніть аналітичний висновок.
        """

        try:
            analysis_result = self.analyst_agent.run(analysis_task)
            results["analysis"] = str(analysis_result)
            print(f"[OK] Analysis завершено")
        except Exception as e:
            print(f"[ERROR] Analysis помилка: {e}")
            results["analysis"] = f"Помилка: {e}"
            return results

        # Агент 3: Reporter
        print("\n" + "="*60)
        print("REPORTER AGENT: Створення звіту")
        print("="*60)

        report_task = f"""
Створіть фінальний звіт про: {topic}

Дослідження: {results["research"]}
Аналіз: {results["analysis"]}

1. Створіть структурований звіт з трьох розділів:
   - Executive Summary
   - Ключові висновки
   - Рекомендації
2. Збережіть звіт через save_memory з ключем "final_report"

Поверніть фінальний звіт.
        """

        try:
            report_result = self.reporter_agent.run(report_task)
            results["report"] = str(report_result)
            print(f"[OK] Report завершено")
        except Exception as e:
            print(f"[ERROR] Report помилка: {e}")
            results["report"] = f"Помилка: {e}"

        return results

    def _demo_mode_sequential(self, topic: str) -> Dict[str, Any]:
        """Демо режим для sequential підходу"""
        print("[WARNING] Працюємо в демо режимі без API")

        # Симулюємо роботу агента
        research = search_web(topic)
        analysis = analyze_text(research)

        report = f"""
╔══════════════════════════════════════════════════════════════╗
║         SMOLAGENTS MULTI-AGENT REPORT (DEMO)                 ║
╚══════════════════════════════════════════════════════════════╝

Дата: {datetime.now():%Y-%m-%d %H:%M:%S}
Тема: {topic}

═══════════════════════════════════════════════════════════════
ЕТАП 1 - RESEARCH
═══════════════════════════════════════════════════════════════

{research}

═══════════════════════════════════════════════════════════════
ЕТАП 2 - ANALYSIS
═══════════════════════════════════════════════════════════════

{analysis}

═══════════════════════════════════════════════════════════════
ЕТАП 3 - REPORT
═══════════════════════════════════════════════════════════════

ВИСНОВКИ:
- AI активно впроваджується в освітній процес
- Основний фокус на персоналізації та автоматизації
- Необхідна підготовка викладацького складу

РЕКОМЕНДАЦІЇ:
- Розробити стратегію впровадження AI
- Інвестувати в навчання персоналу
- Забезпечити етичні стандарти

[OK] Дослідження завершено (демо режим)
        """

        save_memory("final_report", report)
        print(report)

        return {
            "approach": "sequential",
            "topic": topic,
            "result": report,
            "mode": "demo"
        }

    def _demo_mode_multi_agent(self, topic: str) -> Dict[str, Any]:
        """Демо режим для multi-agent підходу"""
        print("[WARNING] Працюємо в демо режимі без API")

        # Етап 1: Research
        print("\nRESEARCHER: Збираю інформацію...")
        research = search_web(topic)
        print(research[:200] + "...")

        # Етап 2: Analysis
        print("\nANALYST: Аналізую дані...")
        analysis = analyze_text(research)
        print(analysis)

        # Етап 3: Report
        print("\nREPORTER: Створюю звіт...")
        report = f"Звіт на основі дослідження та аналізу створено."

        return {
            "approach": "multi-agent",
            "topic": topic,
            "research": research,
            "analysis": analysis,
            "report": report,
            "mode": "demo"
        }

# ===========================
# ГОЛОВНА ФУНКЦІЯ
# ===========================

def main():
    """Демонстрація SmolAgents мультиагентної системи"""

    print("""
╔══════════════════════════════════════════════════════════════╗
║     SMOLAGENTS MULTI-AGENT SYSTEM                            ║
║     Два підходи до мультиагентності                          ║
╚══════════════════════════════════════════════════════════════╝
    """)

    # Перевірка пакетів
    print("\nПеревірка пакетів:")
    try:
        import smolagents
        print(f"   [OK] SmolAgents: встановлено")
    except:
        print("   [ERROR] SmolAgents: не встановлено")

    # API ключ
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key:
        print(f"   [OK] API ключ: {api_key[:10]}...{api_key[-4:]}")
    else:
        print("   [WARNING] API ключ: не знайдено (демо режим)")

    # Створення системи
    system = SmolAgentsMultiAgentSystem(model_type="openai", api_key=api_key)

    # Тема дослідження
    topic = "Штучний інтелект в освіті України 2025: можливості та виклики"

    # Демонстрація обох підходів
    print("\n" + "="*60)
    print("Оберіть підхід:")
    print("1. Sequential (один агент, три етапи)")
    print("2. Multi-Agent (три агенти)")
    print("="*60)

    choice = input("\nВаш вибір (1 або 2, Enter для demo): ").strip()

    if choice == "1":
        result = system.run_sequential(topic)
    elif choice == "2":
        result = system.run_multi_agent(topic)
    else:
        # Демо обох підходів
        print("\nДемо Sequential підходу:")
        result1 = system.run_sequential(topic)

        print("\n\nДемо Multi-Agent підходу:")
        result2 = system.run_multi_agent(topic)

        result = {"sequential": result1, "multi_agent": result2}

    # Збереження результатів
    filename = f"smolagents_multiagent_{datetime.now():%Y%m%d_%H%M%S}.json"
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"\n\nРезультати збережено: {filename}")
    print(f"Пам'ять: smolagents_multiagent_memory.json")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nПрограму перервано")
    except Exception as e:
        print(f"\nКритична помилка: {e}")
        import traceback
        traceback.print_exc()
