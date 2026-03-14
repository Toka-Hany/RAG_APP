from rag.routing.query_analyzer import QueryAnalyzer

analyzer = QueryAnalyzer()

print(analyzer.analyze("What is diversification?"))
print(analyzer.analyze("What was the revenue in 2020?"))
print(analyzer.analyze("What is Tesla stock price?"))
print(analyzer.analyze("Should I invest in tech stocks?"))