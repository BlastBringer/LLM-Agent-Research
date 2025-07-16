from crewai import Agent, LLM ,Crew,Process,Task
from crewai_tools import SerperDevTool
from crewai.project import CrewBase, agent , crew, task
from typing import List
import yaml

agents_config = "config/agents.yaml"
task_config = "config/tasks.yaml"
with open(agents_config, "r") as file:
    agents_yaml = yaml.safe_load(file)

with open(task_config, "r") as file:
    tasks_yaml = yaml.safe_load(file)

llm = LLM(
    model="ollama/mistral:latest",
    base_url="http://localhost:11434",
    stream=True,
)


interpreter = Agent(
    config=agents_yaml['interpreter'],
    llm=llm,
    verbose=True
)

classifier = Agent(
    config=agents_yaml['type_classifier'],
    llm=llm,
    verbose=True
)

tool_selector = Agent(
    config=agents_yaml['tool_selector'],
    llm=llm,
    verbose=True
)


result_aggregator = Agent(
    config=agents_yaml['result_aggregator'],
    llm=llm,
    verbose=True
)

sympy_executor = Agent(
    config=agents_yaml['sympy_executor'],
    llm=llm,
    verbose=True
)

numeric_executor = Agent(
    config=agents_yaml['numeric_executor'],
    llm=llm,
    verbose=True
)
search_agent = Agent(
    config=agents_yaml['search_agent'],
    llm=llm,
    verbose=True
)
verifier = Agent(
    config=agents_yaml['verifier'],
    llm=llm,
    verbose=True
)
output_cleaner = Agent(
    config=agents_yaml['output_cleaner'],
    llm=llm,
    verbose=True
)



parse_and_classify = Task(
    config=tasks_yaml['parse_and_classify_task'],
    agent=interpreter,
    verbose=True
)

type_classifier = Task(
    config=tasks_yaml['classify_task_type'],
    agent=classifier,
    verbose=True,
    context=[parse_and_classify]
)

select_tool_task = Task(
    config=tasks_yaml['select_tool_task'],
    agent=tool_selector,
    verbose=True,
    context=[parse_and_classify]
)
execute_symbolic_task = Task(
    config=tasks_yaml['execute_symbolic_task'],
    agent=sympy_executor,
    verbose=True,
    context=[parse_and_classify, type_classifier, select_tool_task]
)

execute_numeric_computation = Task(
    config=tasks_yaml['execute_numeric_computation'],
    agent=numeric_executor,
    verbose=True,
    context=[select_tool_task]
)
search_math_knowledge = Task(
    config=tasks_yaml['search_math_knowledge'],
    agent=search_agent,
    verbose=True,
    context=[select_tool_task]
)
verify_result = Task(
    config=tasks_yaml['verify_result'],
    agent=verifier,
    verbose=True,
    context=[execute_numeric_computation,execute_symbolic_task,search_math_knowledge]
)

clean_output = Task(
    config=tasks_yaml['clean_output'],
    agent=output_cleaner,
    verbose=True,
    context=[verify_result]
)

aggregate_results_task = Task(
    config=tasks_yaml['aggregate_results_task'],
    agent=result_aggregator,
    verbose=True,
    context=[parse_and_classify, select_tool_task, type_classifier,execute_symbolic_task,execute_numeric_computation,search_math_knowledge,verify_result,clean_output]
)


crew = Crew(
    agents=[interpreter,classifier,tool_selector,sympy_executor, numeric_executor, search_agent,verifier,output_cleaner,result_aggregator],
    tasks=[parse_and_classify, type_classifier, select_tool_task,execute_symbolic_task,execute_numeric_computation,search_math_knowledge,verify_result,clean_output,aggregate_results_task],
    process=Process.sequential,
    verbose=True
)



result = crew.kickoff(inputs={"input":"Can you help me simplify this algebraic expression: (3x + 2x) - (x - 4)?"})
print(result)

