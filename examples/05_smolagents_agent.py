"""
Модуль 1: AI Research Agent на SmolAgents
Той самий агент-дослідник для порівняння з LangChain та CrewAI
"""

import os
from typing import Dict, List, Any, Optional
from smolagents import CodeAgent, tool, ApiModel, OpenAIServerModel
from datetime import datetime
import json
import requests

# ===========================
# БАЗОВИЙ АГЕНТ-ДОСЛІДНИК
# ===========================

class SmolAgentsResearchAgent:
    """
    Агент-дослідник на SmolAgents
    Мінімалістичний підхід з фокусом на код
    """
    
    def __init__(self, model_type: str = "openai", api_key: str = None):
        """
        Ініціалізація агента
        model_type: "openai", "hf", або "local"
        """
        # Вибір моделі
        self.model = self._setup_model(model_type, api_key)
        
        # Створення інструментів
        self.tools = self._create_tools()
        
        # Створення агента (CodeAgent за замовчуванням)
        self.agent = self._create_agent()
    
    def _setup_model(self, model_type: str, api_key: str):
        """Налаштування моделі"""
        if model_type == "openai":
            # Використання OpenAI через сервер API
            return OpenAIServerModel(
                model_id="gpt-4",
                api_key=api_key or os.getenv("OPENAI_API_KEY")
            )
        elif model_type == "hf":
            # Використання моделі з Hugging Face Hub
            return ApiModel(
                model_id="meta-llama/Llama-3.3-70B-Instruct",
                token=os.getenv("HF_TOKEN")
            )

    
    def _create_tools(self) -> List:
        """Створення набору інструментів для дослідження"""
        
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
                # Спрощений пошук через DuckDuckGo HTML версію
                url = f"https://duckduckgo.com/html/?q={query}"
                response = requests.get(url)
                
                if response.status_code == 200:
                    # Простий парсинг результатів
                    text = response.text
                    results = []
                    
                    # Виділяємо перші кілька результатів
                    lines = text.split('\n')
                    for line in lines[:100]:
                        if 'result__snippet' in line:
                            # Очищуємо HTML
                            clean_line = line.strip().replace('<b>', '').replace('</b>', '')
                            if len(clean_line) > 50:
                                results.append(clean_line[:200])
                    
                    if results:
                        return "Результати пошуку:\n" + "\n".join(results[:3])
                
                return f"Пошук '{query}' виконано (симуляція)"
            except Exception as e:
                return f"Помилка пошуку: {e}"
        
        @tool
        def get_current_date() -> str:
            """
            Отримати поточну дату та час.
            
            Returns:
                Поточна дата та час у форматі ISO
            """
            return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        @tool
        def analyze_sentiment(text: str) -> Dict[str, Any]:
            """
            Аналіз тональності тексту.
            
            Args:
                text: Текст для аналізу
            
            Returns:
                Словник з результатами аналізу
            """
            # Простий аналіз на основі ключових слів
            positive_indicators = ["успіх", "інновація", "прогрес", "покращення", "розвиток"]
            negative_indicators = ["проблема", "виклик", "ризик", "загроза", "складність"]
            
            text_lower = text.lower()
            
            positive_score = sum(1 for word in positive_indicators if word in text_lower)
            negative_score = sum(1 for word in negative_indicators if word in text_lower)
            
            total = positive_score + negative_score
            if total == 0:
                sentiment = "нейтральна"
                confidence = 0.5
            else:
                if positive_score > negative_score:
                    sentiment = "позитивна"
                    confidence = positive_score / total
                elif negative_score > positive_score:
                    sentiment = "негативна"
                    confidence = negative_score / total
                else:
                    sentiment = "змішана"
                    confidence = 0.5
            
            return {
                "sentiment": sentiment,
                "confidence": round(confidence, 2),
                "positive_indicators": positive_score,
                "negative_indicators": negative_score
            }
        
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
            memory_file = "smolagents_memory.json"
            
            try:
                with open(memory_file, 'r') as f:
                    memory = json.load(f)
            except (FileNotFoundError, json.JSONDecodeError):
                memory = {}
            
            memory[key] = {
                "value": value,
                "timestamp": datetime.now().isoformat()
            }
            
            with open(memory_file, 'w') as f:
                json.dump(memory, f, ensure_ascii=False, indent=2)
            
            return f"Збережено: {key} = {value[:50]}..."
        
        return [search_web, get_current_date, analyze_sentiment, save_memory]
    
    def _create_agent(self):
        """Створення агента"""
        # SmolAgents підтримує два типи агентів
        # CodeAgent - генерує Python код для вирішення задач
        
        instructions_text = """Ви - професійний агент-дослідник. 
            Використовуйте доступні інструменти для збору та аналізу інформації.
            Завжди перевіряйте факти та надавайте структуровані висновки."""
        
        agent = CodeAgent(
            tools=self.tools,
            model=self.model,
            max_steps=5,
            instructions=instructions_text
        )
        
        return agent
    
    def research(self, topic: str) -> Dict[str, Any]:
        """Виконати дослідження на задану тему"""
        print(f"\nSmolAgents: Починаю дослідження теми '{topic}'")
        print("=" * 60)
        
        # Формування задачі для агента
        task = f"""
        Проведіть дослідження на тему: {topic}
        
        Виконайте наступні кроки:
        1. Отримайте поточну дату для часової мітки
        2. Знайдіть інформацію через веб-пошук
        3. Проаналізуйте тональність знайденої інформації
        4. Збережіть ключові факти в пам'ять
        5. Сформуйте структурований висновок
        
        Поверніть детальний звіт.
        """
        
        try:
            # Виконання дослідження
            result = self.agent.run(task)
            
            return {
                "topic": topic,
                "result": str(result),
                "agent_type": "CodeAgent",
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                "topic": topic,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }

# ===========================
# ДЕМОНСТРАЦІЯ
# ===========================

def main():
    """Демонстрація роботи агентів SmolAgents"""
    
    research_topic = "Штучний інтелект в освіті 2025"
    
    print("\nSMOLAGENTS DEMO")
    print("=" * 60)
    
    agent = SmolAgentsResearchAgent(model_type="openai")
    result = agent.research(research_topic)
    
    print(f"\nРезультат:")
    print(f"Тема: {result['topic']}")
    print(f"Тип агента: {result.get('agent_type', 'N/A')}")
    print(f"Результат: {result.get('result', 'Немає результату')[:300]}...")

if __name__ == "__main__":
    # os.environ["OPENAI_API_KEY"] = "your-key-here"
    # os.environ["HF_TOKEN"] = "your-hf-token"
    
    try:
        main()
    except Exception as e:
        print(f"[ERROR] Помилка: {e}")
        print("\nПереконайтеся, що:")
        print("1. Встановлено smolagents: pip install smolagents")
        print("2. Налаштовано API ключі або локальну модель")
