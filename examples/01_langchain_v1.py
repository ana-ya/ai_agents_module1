"""
Модуль 1: AI Research Agent на LangChain 1.0
"""

import os
import sys
from datetime import datetime
import json
from typing import Dict, List, Any
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Setup logging first
from src import setup_logging, get_logger, LoggerMixin
from src.exceptions import APIKeyError, ModelError, ToolError, FileOperationError, ResearchError
from src.error_handling import retry_on_error

# Setup logging
logger = setup_logging(level=os.getenv("LOG_LEVEL", "INFO"))

# Завантаження .env
try:
    from dotenv import load_dotenv
    load_dotenv()
    logger.info(".env файл завантажено")
except ImportError:
    logger.warning("python-dotenv не встановлено, використовуються змінні оточення")

# Імпорти для LangChain 1.0
try:
    from langchain_openai import ChatOpenAI
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import ChatPromptTemplate
    logger.info("LangChain 1.0 компоненти завантажено")
except ImportError as e:
    logger.error(f"Помилка імпорту LangChain: {e}")
    logger.error("Встановіть: pip install langchain-openai langchain-core")
    sys.exit(1)

# ===========================
# LANGCHAIN 1.0 AGENT
# ===========================

class LangChain1Agent(LoggerMixin):
    """
    Агент-дослідник на LangChain 1.0
    Використовує нову архітектуру LCEL (LangChain Expression Language)
    """
    
    def __init__(self, api_key: str = None):
        """Ініціалізація агента"""
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        
        if not self.api_key:
            self.logger.warning("OPENAI_API_KEY не знайдено! AI аналіз буде недоступний")
            self.llm = None
        else:
            try:
                # Створення LLM для LangChain 1.0
                self.llm = ChatOpenAI(
                    model="gpt-4",
                    temperature=0.7,
                    api_key=self.api_key
                )
                self.logger.info("ChatOpenAI LLM створено успішно")
            except Exception as e:
                self.logger.error(f"Помилка створення LLM: {e}", exc_info=True)
                raise ModelError(f"Не вдалося створити LLM: {e}") from e
        
        # Створення інструментів
        self.tools = self._create_tools()
        self.logger.info(f"Створено {len(self.tools)} інструментів")
        
        # Створення ланцюгів (chains) - нова архітектура LangChain 1.0
        self.chains = self._create_chains()
        self.logger.info(f"Створено {len(self.chains)} ланцюгів")
    
    def _create_tools(self) -> Dict:
        """Створення інструментів для дослідження"""
        
        @retry_on_error(max_retries=2, delay=1.0)
        def search_web(query: str) -> str:
            """Пошук інформації в інтернеті"""
            try:
                from ddgs import DDGS
                with DDGS() as ddgs:
                    results = list(ddgs.text(query, max_results=3))
                    if results:
                        output = "Результати пошуку:\n\n"
                        for i, r in enumerate(results, 1):
                            output += f"{i}. {r.get('title', '')}\n"
                            output += f"   {r.get('body', '')[:200]}...\n"
                            output += f"   {r.get('href', '')}\n\n"
                        return output
            except ImportError:
                # Fallback to demo results if ddgs not available
                pass
            except Exception as e:
                raise ToolError(f"Помилка веб-пошуку: {e}") from e
            
            # Демо результат
            return f"""
Демо результати для '{query}':

1. AI трансформує освіту через персоналізацію
   Штучний інтелект дозволяє створювати індивідуальні навчальні траєкторії...

2. Статистика 2025: 85% закладів використовують AI
   За даними дослідження, більшість навчальних закладів впровадили AI-рішення...

3. Виклики впровадження AI в освіті
   Основні проблеми: підготовка кадрів, етичні питання, доступність...
"""
        
        def analyze_data(text: str) -> str:
            """Аналіз даних"""
            word_count = len(text.split())
            sentences = text.count('.') + text.count('!') + text.count('?')
            
            # Простий sentiment аналіз
            positive = ["успіх", "покращення", "інновація", "прогрес", "ефективність"]
            negative = ["проблема", "виклик", "ризик", "загроза", "складність"]
            
            pos_count = sum(1 for word in positive if word in text.lower())
            neg_count = sum(1 for word in negative if word in text.lower())
            
            sentiment = "позитивний" if pos_count > neg_count else "негативний" if neg_count > pos_count else "нейтральний"
            
            return f"""
Аналіз даних:
- Слів: {word_count}
- Речень: {sentences}
- Тональність: {sentiment}
- Позитивних маркерів: {pos_count}
- Негативних маркерів: {neg_count}
"""
        
        def save_to_memory(data: dict) -> str:
            """Збереження в пам'ять"""
            filename = "langchain1_memory.json"
            try:
                with open(filename, 'r', encoding='utf-8') as f:
                    memory = json.load(f)
            except FileNotFoundError:
                memory = {"sessions": []}
            except json.JSONDecodeError:
                # If JSON is corrupted, start fresh
                memory = {"sessions": []}
            
            memory["sessions"].append(data)
            
            try:
                with open(filename, 'w', encoding='utf-8') as f:
                    json.dump(memory, f, ensure_ascii=False, indent=2)
                return f"Збережено в {filename}"
            except Exception as e:
                raise FileOperationError(f"Не вдалося зберегти дані: {e}") from e
        
        return {
            "search_web": search_web,
            "analyze_data": analyze_data,
            "save_to_memory": save_to_memory
        }
    
    def _create_chains(self):
        """Створення ланцюгів для LangChain 1.0 LCEL"""
        chains = {}
        
        if self.llm:
            # Промпт для дослідження
            research_prompt = ChatPromptTemplate.from_messages([
                ("system", "Ви - професійний дослідник AI в освіті. Аналізуйте надану інформацію та створіть структурований звіт."),
                ("human", "Тема: {topic}\n\nДані:\n{data}\n\nСтворіть детальний аналіз.")
            ])
            
            # Ланцюг дослідження (LCEL синтаксис)
            chains["research"] = research_prompt | self.llm | StrOutputParser()
            
            # Ланцюг для висновків
            conclusion_prompt = ChatPromptTemplate.from_messages([
                ("system", "Ви експерт з формування висновків. Узагальніть інформацію."),
                ("human", "{analysis}")
            ])
            
            chains["conclusion"] = conclusion_prompt | self.llm | StrOutputParser()
        
        return chains
    
    async def research_async(self, topic: str) -> dict:
        """Асинхронне дослідження (нова функція LangChain 1.0)"""
        # LangChain 1.0 підтримує async операції
        pass
    
    def research(self, topic: str) -> dict:
        """Синхронне дослідження"""
        if not topic or not topic.strip():
            raise ResearchError("Тема дослідження не може бути порожньою")
        
        # Validate topic length
        if len(topic) > 500:
            raise ResearchError("Тема дослідження занадто довга (максимум 500 символів)")
        
        self.logger.info(f"Початок дослідження теми: {topic}")
        results = {"topic": topic, "timestamp": datetime.now().isoformat()}
        
        try:
            # Крок 1: Пошук
            self.logger.info("Крок 1: Пошук інформації...")
            try:
                search_results = self.tools["search_web"](topic)
            except Exception as e:
                self.logger.warning(f"Помилка пошуку: {e}, використовую демо-результати")
                search_results = f"Результати пошуку недоступні: {e}"
            results["search"] = search_results
            self.logger.info("Крок 1 завершено: пошук інформації")
            
            # Крок 2: Аналіз
            self.logger.info("Крок 2: Аналіз даних...")
            try:
                analysis = self.tools["analyze_data"](search_results)
            except Exception as e:
                self.logger.warning(f"Помилка аналізу: {e}")
                analysis = "Аналіз даних недоступний"
            results["analysis"] = analysis
            self.logger.info("Крок 2 завершено: аналіз даних")
            
            # Крок 3: AI обробка (якщо доступна)
            if self.llm and "research" in self.chains:
                self.logger.info("Крок 3: AI аналіз...")
                try:
                    ai_analysis = self.chains["research"].invoke({
                        "topic": topic,
                        "data": search_results
                    })
                    results["ai_analysis"] = ai_analysis
                    self.logger.info("Крок 3 завершено: AI аналіз")
                except Exception as e:
                    self.logger.error(f"Помилка AI аналізу: {e}", exc_info=True)
                    results["ai_analysis"] = "AI аналіз недоступний"
            else:
                self.logger.info("Крок 3: Демо аналіз (LLM недоступний)...")
                results["ai_analysis"] = self._demo_analysis(topic)
                self.logger.info("Крок 3 завершено: демо аналіз")
            
            # Крок 4: Збереження
            self.logger.info("Крок 4: Збереження результатів...")
            try:
                save_result = self.tools["save_to_memory"](results)
                self.logger.info(f"Крок 4 завершено: {save_result}")
            except Exception as e:
                self.logger.warning(f"Помилка збереження: {e}")
            
            # Створення звіту
            report = self._create_report(results)
            results["report"] = report
            
            # Збереження фінального звіту
            try:
                with open("langchain1_report.json", "w", encoding="utf-8") as f:
                    json.dump(results, f, ensure_ascii=False, indent=2)
                self.logger.info("Звіт збережено: langchain1_report.json")
            except Exception as e:
                self.logger.error(f"Помилка збереження звіту: {e}", exc_info=True)
            
            self.logger.info("Дослідження завершено успішно")
            return results
            
        except Exception as e:
            self.logger.error(f"Помилка під час дослідження: {e}", exc_info=True)
            raise ResearchError(f"Помилка дослідження: {e}") from e
    
    def _demo_analysis(self, topic: str) -> str:
        """Демо аналіз для випадків без API"""
        return f"""
Аналіз теми '{topic}':

**Основні тренди:**
- Персоналізація навчання через AI
- Автоматизація рутинних задач
- Адаптивні навчальні системи

**Переваги:**
- Підвищення ефективності навчання на 30%
- Доступність 24/7
- Індивідуальний підхід

**Виклики:**
- Необхідність підготовки викладачів
- Питання етики та приватності
- Цифрова нерівність

**Прогноз:**
Очікується зростання ринку EdTech на 45% до 2026 року.
"""
    
    def _create_report(self, results: dict) -> str:
        """Створення фінального звіту"""
        return f"""
╔══════════════════════════════════════════════════════════════╗
║              LANGCHAIN 1.0 RESEARCH REPORT                   ║
╚══════════════════════════════════════════════════════════════╝

Дата: {results['timestamp']}
Тема: {results['topic']}
Платформа: LangChain 1.0 + OpenAI GPT-4

════════════════════════════════════════════════════════════════

РЕЗУЛЬТАТИ ПОШУКУ:
{results.get('search', 'Немає даних')}

════════════════════════════════════════════════════════════════

СТАТИСТИЧНИЙ АНАЛІЗ:
{results.get('analysis', 'Немає даних')}

════════════════════════════════════════════════════════════════

AI АНАЛІТИКА:
{results.get('ai_analysis', 'Немає даних')}

════════════════════════════════════════════════════════════════

Дослідження завершено успішно
"""

# ===========================
# ГОЛОВНА ФУНКЦІЯ
# ===========================

def main():
    """Запуск LangChain 1.0 агента"""
    
    logger.info("=" * 60)
    logger.info("LANGCHAIN 1.0 RESEARCH AGENT")
    logger.info("=" * 60)
    
    # Перевірка версій
    logger.info("Перевірка версій пакетів:")
    try:
        import langchain
        logger.info(f"   LangChain: {langchain.__version__}")
    except Exception as e:
        logger.warning(f"   LangChain: не встановлено ({e})")
    
    try:
        import langchain_openai
        logger.info("   LangChain-OpenAI: встановлено")
    except Exception as e:
        logger.warning(f"   LangChain-OpenAI: не встановлено ({e})")
    
    try:
        import openai
        version = getattr(openai, '__version__', 'версія невідома')
        logger.info(f"   OpenAI: {version}")
    except Exception as e:
        logger.warning(f"   OpenAI: не встановлено ({e})")
    
    # Перевірка API ключа
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key:
        logger.info("   API ключ: знайдено")
    else:
        logger.warning("   API ключ: не знайдено")
    
    logger.info("=" * 60)
    
    try:
        # Створення агента
        agent = LangChain1Agent(api_key)
        
        # Дослідження
        topic = "Штучний інтелект в освіті 2025: найновіші тренди"
        result = agent.research(topic)
        
        # Виведення звіту
        if result and "report" in result:
            logger.info("=" * 60)
            logger.info("ЗВІТ СТВОРЕНО")
            logger.info("=" * 60)
            
        logger.info("Дослідження завершено. Перегляньте файли:")
        logger.info("   - langchain1_report.json - повні дані")
        logger.info("   - langchain1_memory.json - збережена історія")
        logger.info("   - logs/ai_agents.log - логи виконання")
        
    except Exception as e:
        logger.error(f"Критична помилка: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.warning("Програму перервано користувачем")
        sys.exit(0)
    except Exception as e:
        logger.critical(f"Критична помилка: {e}", exc_info=True)
        sys.exit(1)
