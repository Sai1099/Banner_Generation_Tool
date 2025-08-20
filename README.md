# AI Banner & Content Generation Tool

An AI-powered tool designed to automate **banner generation** and **content creation** for marketing campaigns.  
This project leverages **LLMs (Gemini 2.5 Pro)** for intelligent tagging and copywriting, along with **Flux Dev 1 Pro API** for banner design automation.

---

## ✨ Features

### 🖼️ Banner Generation
1. **Data-Driven Insights**  
   - Uses historical banner data (URLs, titles, descriptions, click rates).  
   - Extracts **detailed tags** for each banner using **Gemini 2.5 Pro**.  

2. **Custom Banner Creation**  
   - Enter banner details and prompts.  
   - Select the **number of banners** and preferred **style**.  
   - Generates high-quality banners via **Flux Dev 1 Pro API**.  

3. **Text Overlay**  
   - Input custom text for overlay.  
   - Tool provides **3 variations** to choose from.  
   - Selected text overlays are applied to the banners with **logically printed overlays**.  

---

### 📝 Content Generation
1. **Business Type Selection**  
   - Choose your business type for tailored campaign content.  

2. **Campaign Reference**  
   - Specify the number of **notification campaigns** as reference.  
   - Optionally, include additional details for refinement.  

3. **CTR-Based Optimization**  
   - Calculates **CTR (Click Through Rate)**:  
     ```
     CTR = (Overall Clicks / Overall Push) * 100
     ```
   - Generates **3 AI-powered content options** optimized for engagement.  

---

## ⚙️ Tech Stack
- **Backend**: Python (FastAPI)  
- **AI Models**: Gemini 2.5 Pro, Flux Dev 1 Pro API  
- **Libraries**: Pandas, NLTK  
- **Frontend**: Streamlit (for interaction)  

---

## 🚀 Workflow
1. Upload historical banner & campaign data.  
2. Extract tags & insights via **Gemini 2.5 Pro**.  
3. Generate banners & apply text overlays with **Flux Dev 1 Pro API**.  
4. Select business type & campaign references.  
5. Generate optimized **content + banners** with CTR insights.  

---

## 📌 Use Cases
- Marketing teams automating campaign design.  
- Businesses seeking faster **A/B testing** of banners.  
- AI-assisted **creative workflow** for ads and notifications.  

---

## 📂 Project Structure
