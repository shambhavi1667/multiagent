import warnings
warnings.filterwarnings("ignore")
import os
from dotenv import load_dotenv
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_agentchat.agents import AssistantAgent
from autogen_core.tools import FunctionTool
from autogen_agentchat.agents import UserProxyAgent

load_dotenv()

GROQ_API_KEY = os.getenv('GROQ_API_KEY')
GROQ_MODEL = os.getenv('GROQ_MODEL', 'openai/gpt-oss-120b')

# Groq exposes an OpenAI-compatible API, so we reuse OpenAIChatCompletionClient.
# model_info is required because AutoGen doesn't know non-OpenAI model names.
def init_client():
    client = OpenAIChatCompletionClient(
        model=GROQ_MODEL,
        base_url="https://api.groq.com/openai/v1",
        api_key=GROQ_API_KEY,
        model_info={
            "vision": False,
            "function_calling": True,
            "json_output": True,
            "structured_output": True,
            "family": "unknown",
        },
        temperature=0,
    )
    return client

def create_schema_agent(client, chain):
    schema_agent = AssistantAgent(name="schema_agent",
                                model_client=client,
                                description="Retrieves database schema information at start of the conversation. Creates stored procedures.",
                                tools=[FunctionTool(name="get_schema_info", func = chain.get_schema_info, description="Retrieves the database schema and saves it"),
                                       FunctionTool(name="execute_query", func= chain.execute_query, description= "executes postgres query on database"),
                                       FunctionTool(name="get_procedure_info", func= chain.get_procedure_info, description= "retrieves stored procedure information from the database")],
                                system_message=(
                                        "You have two main roles: providing database schema information and creating stored procedures."
                                        "Only use 'get_schema_info' to retrieve schema information and store it. And always provide schema information when you start first."
                                        "You can use 'get_procedure_info' to retrieve stored procedure information from the database."
                                        "You can run CREATE queries on the database to create stored procedure using 'execute_query' function. Ask for user approval before running CREATE query."
                                        "If you could not retrieve the schema information, say 'I failed to get the schema information'"
                                    )
                                ) 
    return schema_agent  



def initiate_planner_agent(client):
    planning_agent = AssistantAgent(
                    name ="planning_agent",
                    description="An agent for planning tasks, this agent should be the first to engage when given a new task.",
                    model_client=client,
                    system_message="""
                    You are a planning agent.
                    Start by calling the schema_agent to retrieve the database schema information. Do not specify any table name.
                    If the schema information is already available in the conversation history, do not call schema_agent.
                    Your team members are:
                        schema_agent: retrieves database schema information and creates stored procedures.
                        customer_agent: accesses and manages customers information and makes updates to the customers table
                        shipment_agent: accesses and manages shipments and products information and makes updates to the product and shipment related tables
                    You only plan and delegate tasks - you do not execute them yourself. 
                    You must ensure each agent completes all steps of their task before moving on to the next step.

                    When assigning tasks, use this format:
                    1. <agent> : <task>
                    
                    Do not bundle tasks together. Each agent should only be assigned one task at a time.

                    """,
                )
    return planning_agent

def create_shipment_agent(client, shipment_chain):
        
    shipment_agent = AssistantAgent(name="shipment_agent",
                                model_client=client,
                                description="Your role is to focus on shipment and products tables.",
                                tools=[FunctionTool(name="execute_query", func= shipment_chain.execute_query, description= "runs postgres query on shipment database"),
                                       FunctionTool(name="exec_send_shipment", func= shipment_chain.exec_send_shipment, description=  "Sends a shipment by executing the 'send_shipment' stored procedure with the provided values.")],
                                system_message=(
                            "You can run queries using 'execute_query' function."
                            "Use the 'exec_send_shipment' function to create a shipment using the 'send_shipment' stored procedure and provided input values. Below is an example of how to provide input values:"
                            "[11, 1, 2, date(2023, 10, 1),[{'product_id': 1, 'quantity': 2}, {'product_id': 2, 'quantity': 3}], 'in transit', 'in transit', 1]"
                            "Only if schema information is available, proceed with the task. If database schema is not available, ask for it to be provided."
                            "Conditions in query should not be case sensitive."
                            "For Insert, Update, and Delete operations, have human to validate the operation before making it."
                            "Ensure all necessary queries affecting multiple tables are executed in the correct order to maintain referential integrity."
                            )
                            )
    return shipment_agent

def create_customer_agent(client, customer_chain):
    customer_agent = AssistantAgent(name="customer_agent",
                                model_client=client,
                                description="Your role is to focus on customer data management.",
                                tools=[FunctionTool(name="exec_add_customer", func= customer_chain.exec_add_customer, description=  "Adds a customer to the table customers in the database."),
                                       FunctionTool(name="execute_query", func= customer_chain.execute_query, description= "runs postgres query on the database")],
                                system_message=(
                                        "Your role is to manage customer information in the database. You can run queries using 'execute_query' function."
                                        "Use 'exec_add_customer' to add a customer to the database using the add_customer stored procedure and prvided input values."
                                        "For Insert, Update, and Delete operations, have human to validate the operation before making it. Ask for user approval before executing these queries"
                                        "Only if schema information is available, proceed with the task. If database schema is not available, ask for it to be provided."
                                        "Conditions in query should not be case sensitive."
                                    )
                        )
    return customer_agent

def get_user_input(dummy_var): # have to pass a dummy variable to match the function signature
    return input("(Type 'bye' to exit):")

def create_user_proxy():
    user_proxy_agent = UserProxyAgent("user_proxy", 
                        description="Get user input",
                        input_func = get_user_input)
    return user_proxy_agent