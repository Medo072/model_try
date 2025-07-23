import os
import glob
import asyncio
import gradio as gr
from typing import Dict, List, Optional, TypedDict
from datetime import datetime
import pdf2image
from PIL import Image
import base64
from io import BytesIO
import json
import fitz

from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, AIMessage
from langchain.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.types import Command  # Import Command
from langgraph.checkpoint.memory import MemorySaver
from IPython.display import display, Image

# State definition for LangGraph
class ChatState(TypedDict):
    messages: List[dict]
    merchant_id: Optional[str]
    month: Optional[str]
    year: Optional[str]
    pdf_path: Optional[str]
    processed_images: Optional[List[str]]  # Base64 encoded images
    current_step: str
    user_input: str

class MerchantAnalyticsChatbot:
    def __init__(self, openrouter_api_key: str, data_directory: str = "./analytics_data"):
        self.api_key = openrouter_api_key
        self.data_directory = data_directory
        self.llm = ChatOpenAI(
            base_url="https://models.github.ai/inference",
            api_key="github_pat_11AT2KJ2Q0lsYNrBx7w4gH_ylqbdXalKwY9bNrcmEX5BypUQ4J7BTQ7B7rNVFwGQRICN",
            model="openai/gpt-4.1-nano"  # You can change this model
        )
        
        # Initialize LangGraph
        self.workflow = self._create_workflow()
        self.memory = MemorySaver()
        self.app = self.workflow.compile(checkpointer=self.memory)
    
    def _create_workflow(self):
        """Create the LangGraph workflow using Commands"""
        workflow = StateGraph(ChatState)
        
        # Add nodes
        workflow.add_node("welcome", self.welcome_node)
        workflow.add_node("collect_merchant_info", self.collect_merchant_info)
        workflow.add_node("collect_month_info", self.collect_month_info)
        workflow.add_node("find_and_process_pdf", self.find_and_process_pdf)
        workflow.add_node("answer_questions", self.answer_questions)
        workflow.add_node("error_handler", self.error_handler)
        
        # Set entry point
        workflow.set_entry_point("welcome")
        
        return workflow
    
    async def welcome_node(self, state: ChatState) -> Command:
        """Welcome message and initial setup"""
        welcome_msg = """
        Hello! I'm your Merchant Analytics Assistant. I can help you analyze your monthly merchant data.
        
        To get started, I'll need:
        1. Your Merchant ID
        2. The month and year you're interested in (e.g., April 2025)
        
        Please provide your Merchant ID first.
        """
        
        state["messages"].append({
            "role": "assistant", 
            "content": welcome_msg.strip()
        })
        state["current_step"] = "collect_merchant_info"
        
        # Return command to go to collect_merchant_info
        return Command(goto="collect_merchant_info", update=state)
    
    async def collect_merchant_info(self, state: ChatState) -> Command:
        """Collect merchant ID from user"""
        user_input = state["user_input"].strip()
        
        # Simple validation for merchant ID
        if len(user_input) > 3:
            state["merchant_id"] = user_input
            state["messages"].append({
                "role": "assistant",
                "content": f"Great! I've recorded your Merchant ID as: {user_input}\n\nNow, please provide the month and year you're interested in (e.g., 'April 2025' or 'Apr 2025')."
            })
            state["current_step"] = "collect_month_info"
            return Command(goto="collect_month_info", update=state)
        else:
            state["messages"].append({
                "role": "assistant",
                "content": "Please provide a valid Merchant ID. It should be more than 3 characters."
            })
            return Command(goto=END, update=state)
    
    async def collect_month_info(self, state: ChatState) -> Command:
        """Collect month and year information"""
        user_input = state["user_input"].strip().lower()
        
        # Parse month and year
        try:
            month_year = self._parse_month_year(user_input)
            if month_year:
                state["month"] = month_year["month"]
                state["year"] = month_year["year"]
                state["messages"].append({
                    "role": "assistant",
                    "content": f"Perfect! Looking for data from {month_year['month']} {month_year['year']}. Let me search for your analytics file..."
                })
                state["current_step"] = "find_and_process_pdf"
                return Command(goto="find_and_process_pdf", update=state)
            else:
                raise ValueError("Invalid format")
        except Exception as e:
            state["messages"].append({
                "role": "assistant",
                "content": "Please provide the month and year in a format like 'April 2025', 'Apr 2025', or '04 2025'."
            })
            return Command(goto=END, update=state)
    
    async def find_and_process_pdf(self, state: ChatState) -> Command:
        """Find and process the PDF file"""
        try:
            # Construct expected filename pattern
            month_abbr = self._get_month_abbreviation(state["month"])
            pattern = f"digest_{month_abbr}_{state['year']}*{state['merchant_id']}*.pdf"
            
            # Search for the file
            search_pattern = os.path.join(self.data_directory, pattern)
            matching_files = glob.glob(search_pattern)
            
            if not matching_files:
                state["current_step"] = "error"
                state["messages"].append({
                    "role": "assistant",
                    "content": f"I couldn't find an analytics file for Merchant ID '{state['merchant_id']}' for {state['month']} {state['year']}. Please check if the merchant ID and date are correct."
                })
                return Command(goto="error_handler", update=state)
            
            pdf_path = matching_files[0]  # Take the first match
            state["pdf_path"] = pdf_path
            
            # Process PDF to images
            state["messages"].append({
                "role": "assistant",
                "content": "Found your analytics file! Processing the document..."
            })
            
            processed_images = await self._process_pdf_to_images(pdf_path)
            state["processed_images"] = processed_images
            
            state["messages"].append({
                "role": "assistant",
                "content": f"Successfully processed your analytics document ({len(processed_images)} pages). Now you can ask me questions about your merchant analytics! For example:\n\n- What were the total sales this month?\n- Show me the transaction trends\n- What were the top-selling products?\n- How did we perform compared to last month?"
            })
            state["current_step"] = "answer_questions"
            return Command(goto="answer_questions", update=state)
            
        except Exception as e:
            state["current_step"] = "error"
            state["messages"].append({
                "role": "assistant",
                "content": f"An error occurred while processing your file: {str(e)}"
            })
            return Command(goto="error_handler", update=state)
    
    async def answer_questions(self, state: ChatState) -> Command:
        """Answer user questions based on processed images"""
        user_question = state["user_input"]
        
        # Skip if no actual user input (initial load)
        if not user_question or user_question == state.get("last_processed_input"):
            return Command(goto=END, update=state)
        
        try:
            # Create a prompt for analyzing the images
            analysis_prompt = f"""
            You are an expert merchant analytics assistant. The user has asked: "{user_question}"
            
            I have processed their merchant analytics document into images. Please analyze these images and provide a comprehensive answer to their question.
            
            Focus on:
            - Specific data points and metrics
            - Trends and patterns
            - Actionable insights
            - Clear, business-focused explanations
            
            If you cannot find specific information in the images, let the user know what information is available and suggest related insights you can provide.
            """
            
            response = await self._generate_response_with_images(analysis_prompt, user_question, state["processed_images"])
            
            state["messages"].append({
                "role": "assistant",
                "content": response
            })
            state["last_processed_input"] = user_question
            
            # Continue in answer_questions mode for follow-up questions
            return Command(goto="answer_questions", update=state)
        except Exception as e:
            state["messages"].append({
                "role": "assistant",
                "content": f"I encountered an error while analyzing your data: {str(e)}. Please try rephrasing your question."
            })
            return Command(goto="answer_questions", update=state)
    
    async def error_handler(self, state: ChatState) -> Command:
        """Handle errors gracefully"""
        state["messages"].append({
            "role": "assistant",
            "content": "I'm sorry, but I encountered an issue. Please try starting over or contact support if the problem persists."
        })
        return Command(goto=END, update=state)
    
    def _parse_month_year(self, input_text: str) -> Optional[Dict[str, str]]:
        """Parse month and year from user input"""
        months = {
            'jan': 'January', 'january': 'January',
            'feb': 'February', 'february': 'February',
            'mar': 'March', 'march': 'March',
            'apr': 'April', 'april': 'April',
            'may': 'May',
            'jun': 'June', 'june': 'June',
            'jul': 'July', 'july': 'July',
            'aug': 'August', 'august': 'August',
            'sep': 'September', 'september': 'September',
            'oct': 'October', 'october': 'October',
            'nov': 'November', 'november': 'November',
            'dec': 'December', 'december': 'December'
        }
        
        # Split input and look for month and year
        parts = input_text.replace(',', ' ').split()
        month = None
        year = None
        
        for part in parts:
            if part.lower() in months:
                month = months[part.lower()]
            elif part.isdigit() and len(part) == 4:
                year = part
            elif part.isdigit() and len(part) <= 2:
                # Handle numeric month
                month_num = int(part)
                if 1 <= month_num <= 12:
                    month_names = ['January', 'February', 'March', 'April', 'May', 'June',
                                 'July', 'August', 'September', 'October', 'November', 'December']
                    month = month_names[month_num - 1]
        
        if month and year:
            return {"month": month, "year": year}
        return None
    
    def _get_month_abbreviation(self, month_name: str) -> str:
        """Get month abbreviation for filename matching"""
        month_abbr = {
            'January': 'Jan', 'February': 'Feb', 'March': 'Mar',
            'April': 'Apr', 'May': 'May', 'June': 'Jun',
            'July': 'Jul', 'August': 'Aug', 'September': 'Sep',
            'October': 'Oct', 'November': 'Nov', 'December': 'Dec'
        }
        return month_abbr.get(month_name, month_name[:3])
    

    async def _process_pdf_to_images(self, pdf_path: str) -> List[str]:
        """Convert PDF pages to base64 encoded images using PyMuPDF"""
        try:
            # Open the PDF document
            pdf_document = fitz.open(pdf_path)
            
            base64_images = []
            
            # Process each page
            for page_num in range(pdf_document.page_count):
                # Get the page
                page = pdf_document[page_num]
                
                # Set the resolution (DPI)
                zoom = 200 / 72  # 200 DPI (72 is the default)
                matrix = fitz.Matrix(zoom, zoom)
                
                # Render page to pixmap
                pixmap = page.get_pixmap(matrix=matrix)
                
                # Convert pixmap to PIL Image
                img_data = pixmap.tobytes("png")
                
                # Convert to base64
                img_base64 = base64.b64encode(img_data).decode()
                base64_images.append(img_base64)
                
                # Clean up pixmap
                pixmap = None
            
            # Close the PDF document
            pdf_document.close()
            
            return base64_images
            
        except Exception as e:
            raise Exception(f"Failed to process PDF: {str(e)}")
    
    async def _generate_response_with_images(self, prompt: str, question: str, images: List[str]) -> str:
        """Generate response using LLM with image analysis"""
        try:
            # Prepare message content with images
            message_content = []
            
            # Add the text prompt
            message_content.append({
                "type": "text",
                "text": f"{prompt}\n\nUser Question: {question}"
            })
            
            # Add each image
            for i, img_base64 in enumerate(images[2:]):
                message_content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{img_base64}",
                        "detail": "high"  # Use "low" for faster/cheaper processing
                    }
                })
            
            # Create the message with images
            messages = [
                HumanMessage(content=message_content)
            ]
            
            # Get response from vision model
            response = await self.llm.ainvoke(messages)
            return response.content
            
        except Exception as e:
            return f"I apologize, but I encountered an error while analyzing your data: {str(e)}. Please try again with a different question."
    
# Gradio Interface
class GradioInterface:
    def __init__(self, chatbot: MerchantAnalyticsChatbot):
        self.chatbot = chatbot
        self.session_states = {}
    
    def get_session_state(self, session_id: str) -> ChatState:
        """Get or create session state"""
        if session_id not in self.session_states:
            self.session_states[session_id] = {
                "messages": [],
                "merchant_id": None,
                "month": None,
                "year": None,
                "pdf_path": None,
                "processed_images": None,
                "current_step": "welcome",
                "user_input": "",
                "last_processed_input": None
            }
        return self.session_states[session_id]
    
    async def chat_response(self, message: str, history: List[List[str]], session_id: str) -> tuple:
        """Handle chat responses"""
        if not message.strip():
            return history, history
        
        # Get session state
        state = self.get_session_state(session_id)
        
        # Add user message to state
        state["user_input"] = message
        state["messages"].append({"role": "user", "content": message})
        
        # Add user message to history
        history.append([message, None])
        
        try:
            # Process through LangGraph
            config = {"configurable": {"thread_id": session_id}}
            
            # Run the graph starting from the current step
            if state["current_step"] == "welcome":
                result = await self.chatbot.app.ainvoke(state, config=config)
            else:
                # For subsequent interactions, we need to determine which node to invoke
                current_node = state["current_step"]
                
                # Manually route to the correct node based on current state
                if current_node == "collect_merchant_info" and not state.get("merchant_id"):
                    command = await self.chatbot.collect_merchant_info(state)
                elif current_node == "collect_month_info" and not state.get("month"):
                    command = await self.chatbot.collect_month_info(state)
                elif current_node == "find_and_process_pdf" and not state.get("processed_images"):
                    command = await self.chatbot.find_and_process_pdf(state)
                elif current_node == "answer_questions":
                    command = await self.chatbot.answer_questions(state)
                else:
                    # Default to answer questions if we have all the data
                    if state.get("processed_images"):
                        command = await self.chatbot.answer_questions(state)
                    else:
                        # Start over if in an unknown state
                        result = await self.chatbot.app.ainvoke(state, config=config)
                
                # Update state from command if we didn't run the full graph
                if 'command' in locals() and hasattr(command, 'update'):
                    state.update(command.update)
                    if hasattr(command, 'goto') and command.goto != END:
                        state["current_step"] = command.goto
                    result = state
                else:
                    result = state
            
            # Update session state
            self.session_states[session_id] = result
            
            # Get the latest assistant message
            assistant_messages = [msg for msg in result["messages"] if msg["role"] == "assistant"]
            if assistant_messages:
                latest_message = assistant_messages[-1]
                history[-1][1] = latest_message["content"]
            
        except Exception as e:
            history[-1][1] = f"An error occurred: {str(e)}"
        
        return history, history
    
    def create_interface(self):
        """Create Gradio interface"""
        with gr.Blocks(title="Merchant Analytics Chatbot", theme=gr.themes.Soft()) as interface:
            gr.Markdown("# 📊 Merchant Analytics Assistant")
            gr.Markdown("Get insights from your monthly merchant analytics reports")
            
            with gr.Row():
                with gr.Column(scale=4):
                    chatbot = gr.Chatbot(
                        height=500,
                        show_label=False,
                        container=False,
                        bubble_full_width=False
                    )
                    
                    with gr.Row():
                        msg = gr.Textbox(
                            placeholder="Type your message here...",
                            show_label=False,
                            scale=4
                        )
                        send_btn = gr.Button("Send", scale=1, variant="primary")
                
                with gr.Column(scale=1):
                    gr.Markdown("### Instructions")
                    gr.Markdown("""
                    1. **Merchant ID**: Provide your merchant identifier
                    2. **Month/Year**: Specify the period (e.g., "April 2025")
                    3. **Ask Questions**: Once your data is loaded, ask about:
                       - Sales performance
                       - Transaction trends  
                       - Customer insights
                       - Product analysis
                    """)
                    
                    clear_btn = gr.Button("Clear Chat", variant="secondary")
            
            # Session state
            session_id = gr.State(lambda: f"session_{datetime.now().timestamp()}")
            
            # Event handlers
            async def respond(message, history, sess_id):
                return await self.chat_response(message, history, sess_id)
            
            def clear_chat(sess_id):
                # Clear the session state
                if sess_id in self.session_states:
                    del self.session_states[sess_id]
                return [], []
            
            # Initialize chat
            async def init_chat(sess_id):
                # Get initial state
                state = self.get_session_state(sess_id)
                
                # Run welcome node
                config = {"configurable": {"thread_id": sess_id}}
                result = await self.chatbot.app.ainvoke(state, config=config)
                self.session_states[sess_id] = result
                
                # Get welcome message
                welcome_messages = [msg for msg in result["messages"] if msg["role"] == "assistant"]
                if welcome_messages:
                    return [[None, welcome_messages[0]["content"]]]
                return []
            
            # Wire up events
            msg.submit(
                respond,
                inputs=[msg, chatbot, session_id],
                outputs=[chatbot, chatbot]
            ).then(
                lambda: "",
                outputs=[msg]
            )
            
            send_btn.click(
                respond,
                inputs=[msg, chatbot, session_id],
                outputs=[chatbot, chatbot]
            ).then(
                lambda: "",
                outputs=[msg]
            )
            
            clear_btn.click(
                clear_chat,
                inputs=[session_id],
                outputs=[chatbot]
            )
            
            # Initialize with welcome message
            interface.load(
                init_chat,
                inputs=[session_id],
                outputs=[chatbot]
            )
        
        return interface

# Main application
def main():
    # Configuration
    OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "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")
    DATA_DIRECTORY = os.getenv("DATA_DIRECTORY", "./analytics_data")
    
    if not OPENROUTER_API_KEY:
        raise ValueError("Please set the OPENROUTER_API_KEY environment variable")
    
    # Create data directory if it doesn't exist
    os.makedirs(DATA_DIRECTORY, exist_ok=True)
    
    # Initialize chatbot
    chatbot = MerchantAnalyticsChatbot(
        openrouter_api_key=OPENROUTER_API_KEY,
        data_directory=DATA_DIRECTORY
    )
    
    # Create Gradio interface
    gradio_interface = GradioInterface(chatbot)
    interface = gradio_interface.create_interface()
    
    # Launch
    interface.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True  # Set to False for local only
    )

if __name__ == "__main__":
    main()