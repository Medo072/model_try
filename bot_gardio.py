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

from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, AIMessage
from langchain.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

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
            base_url="https://openrouter.ai/api/v1",
            api_key=openrouter_api_key,
            model="openai/gpt-4.1-nano"  # You can change this model
        )
    
    async def process_message(self, state: ChatState) -> ChatState:
        """Process a message based on current state"""
        try:
            current_step = state.get("current_step", "welcome")
            
            if current_step == "welcome":
                return await self.welcome_node(state)
            elif current_step == "collect_merchant" or not state.get("merchant_id"):
                return await self.collect_merchant_info(state)
            elif current_step == "collect_month" or not state.get("month"):
                return await self.collect_month_info(state)
            elif current_step == "find_pdf" or not state.get("processed_images"):
                return await self.find_and_process_pdf(state)
            else:
                return await self.answer_questions(state)
                
        except Exception as e:
            state["current_step"] = "error"
            state["messages"].append({
                "role": "assistant",
                "content": f"I encountered an error: {str(e)}. Please try again or restart the conversation."
            })
            return state
    
    async def welcome_node(self, state: ChatState) -> ChatState:
        """Welcome message and initial setup"""
        if not state["messages"]:  # Only show welcome once
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
        
        state["current_step"] = "collect_merchant"
        return state
    
    async def collect_merchant_info(self, state: ChatState) -> ChatState:
        """Collect merchant ID from user"""
        user_input = state["user_input"].strip()
        
        # Simple validation for merchant ID (you can make this more sophisticated)
        if len(user_input) > 3:
            state["merchant_id"] = user_input
            state["messages"].append({
                "role": "assistant",
                "content": f"Great! I've recorded your Merchant ID as: {user_input}\n\nNow, please provide the month and year you're interested in (e.g., 'April 2025' or 'Apr 2025')."
            })
            state["current_step"] = "collect_month"
        else:
            state["messages"].append({
                "role": "assistant",
                "content": "Please provide a valid Merchant ID. It should be more than 3 characters."
            })
            state["current_step"] = "collect_merchant"
        
        return state
    
    async def collect_month_info(self, state: ChatState) -> ChatState:
        """Collect month and year information"""
        user_input = state["user_input"].strip().lower()
        
        # Parse month and year
        try:
            # Handle different formats
            month_year = self._parse_month_year(user_input)
            if month_year:
                state["month"] = month_year["month"]
                state["year"] = month_year["year"]
                state["messages"].append({
                    "role": "assistant",
                    "content": f"Perfect! Looking for data from {month_year['month']} {month_year['year']}. Let me search for your analytics file..."
                })
                state["current_step"] = "find_pdf"
            else:
                raise ValueError("Invalid format")
        except:
            state["messages"].append({
                "role": "assistant",
                "content": "Please provide the month and year in a format like 'April 2025', 'Apr 2025', or '04 2025'."
            })
        
        return state
    
    async def find_and_process_pdf(self, state: ChatState) -> ChatState:
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
                return state
            
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
            state["current_step"] = "answer"
            
        except Exception as e:
            state["current_step"] = "error"
            state["messages"].append({
                "role": "assistant",
                "content": f"An error occurred while processing your file: {str(e)}"
            })
        
        return state
    
    async def answer_questions(self, state: ChatState) -> ChatState:
        """Answer user questions based on processed images"""
        user_question = state["user_input"]
        
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
            
            # For this example, we'll use a text-based response
            # In a real implementation, you'd send the images to a vision-capable model
            response = await self._generate_response_with_images(analysis_prompt, user_question, state["processed_images"])
            
            state["messages"].append({
                "role": "assistant",
                "content": response
            })
            
        except Exception as e:
            state["messages"].append({
                "role": "assistant",
                "content": f"I encountered an error while analyzing your data: {str(e)}. Please try rephrasing your question."
            })
        
        return state
    
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
        """Convert PDF pages to base64 encoded images"""
        try:
            # Convert PDF to images
            images = pdf2image.convert_from_path(pdf_path, dpi=200)
            
            base64_images = []
            for i, image in enumerate(images):
                # Convert PIL image to base64
                buffer = BytesIO()
                image.save(buffer, format="PNG")
                img_base64 = base64.b64encode(buffer.getvalue()).decode()
                base64_images.append(img_base64)
            
            return base64_images
            
        except Exception as e:
            raise Exception(f"Failed to process PDF: {str(e)}")
    
    async def _generate_response_with_images(self, prompt: str, question: str, images: List[str]) -> str:
        """OpenAI-specific format for image analysis"""
        
        try:
            messages = [{
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"""
                        {prompt}
                        
                        User Question: {question}
                        
                        Please analyze these merchant analytics images and provide specific insights based on the data shown.
                        """
                    }
                ] + [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{base64_image}",
                            "detail": "high"
                        }
                    } for base64_image in images
                ]
            }]
            
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
                "user_input": ""
            }
        return self.session_states[session_id]
    
    async def chat_response(self, message: str, history: List[List[str]], session_id: str) -> tuple:
        """Handle chat responses"""
        if not message.strip():
            return history, history
        
        # Get session state
        state = self.get_session_state(session_id)
        state["user_input"] = message
        
        # Add user message to history
        history.append([message, None])
        
        try:
            # Process message through simplified chatbot
            result = await self.chatbot.process_message(state)
            
            # Update session state
            self.session_states[session_id] = result
            
            # Get the latest assistant message
            if result["messages"]:
                latest_message = result["messages"][-1]
                if latest_message["role"] == "assistant":
                    history[-1][1] = latest_message["content"]
            
        except Exception as e:
            import traceback
            print(f"Error: {e}")
            print(traceback.format_exc())
            history[-1][1] = f"An error occurred: {str(e)}. Please try again."
        
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
            
            def clear_chat():
                return [], []
            
            # Initialize chat
            def init_chat():
                initial_state = {
                    "messages": [],
                    "merchant_id": None,
                    "month": None,
                    "year": None,
                    "pdf_path": None,
                    "processed_images": None,
                    "current_step": "welcome",
                    "user_input": ""
                }
                
                # Trigger welcome message
                welcome_msg = """
                Hello! I'm your Merchant Analytics Assistant. I can help you analyze your monthly merchant data.
                
                To get started, I'll need:
                1. Your Merchant ID  
                2. The month and year you're interested in (e.g., April 2025)
                
                Please provide your Merchant ID first.
                """
                
                return [[None, welcome_msg.strip()]]
            
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
            
            clear_btn.click(clear_chat, outputs=[chatbot])
            
            # Initialize with welcome message
            interface.load(init_chat, outputs=[chatbot])
        
        return interface

# Main application
def main():
    # Configuration
    OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "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")
    DATA_DIRECTORY = os.getenv("DATA_DIRECTORY", "./analytics_data")
    
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