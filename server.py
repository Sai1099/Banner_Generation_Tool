
import streamlit as st
import pandas as pd
import numpy as np
import tempfile
import os
from langchain_google_genai import ChatGoogleGenerativeAI
import json
import requests
from PIL import Image,ImageOps,ImageDraw,ImageFont
from io import BytesIO
import sys
import pysqlite3
sys.modules['sqlite3'] = pysqlite3
import chromadb
from sentence_transformers import SentenceTransformer
from gradio_client import Client,handle_file
import threading
import io
import time

chromadb.api.client.SharedSystemClient.clear_system_cache()

st.set_page_config(page_title="Banner Creation Tool",page_icon="main_logo.png",layout="wide")
#------------------------------------------------------------------------------------------------------------

#--------------------------------------------
# ----------------------------------------------------------------


if "main_json_file" not in st.session_state:
     st.session_state.main_json_file = None
if "img_btn_clicked" not in st.session_state:
    st.session_state.img_btn_clicked = False
if "img_already_generated" not in st.session_state:
    st.session_state.img_already_generated = False


data_driv_df = pd.read_csv("banner_data.csv",encoding='latin-1')
main_file = pd.read_csv("banner_data.csv",encoding='latin-1')
#Select * from data_deiv_df 
main_top_df = (
    data_driv_df.sort_values(by="Total clicks", ascending=False)
)

st.title("Generate Data Driven Banners using AI")
d_lef,d_rig = st.columns(2)
with d_lef:
 st.subheader("Enter_Credentials")
 API_KEY = st.text_input("Enter your GOOGLE_API_KEYS to continue generation:",type="password")
 
 if st.subheader("Tags"):
  data_bussiness = data_driv_df["Suggested Banner Type"].unique().tolist()
  biz_sel = st.selectbox("Choose the Bussiness:",options=data_bussiness)

  optional_prompt  = st.text_input("Additional Input to generate the tags")

  if st.button("Generate Tags Based on Previous Data") or  optional_prompt:
      
      main_d = data_driv_df[data_driv_df["Suggested Banner Type"] == biz_sel]

      if "main_d" not in st.session_state:
          st.session_state["main_d"] = main_d      
     

      asa = main_d["tags"].to_list()
      if API_KEY:
        os.environ["GOOGLE_API_KEY"] = API_KEY



        
        llm = ChatGoogleGenerativeAI(
                                        model="gemini-2.0-flash",
                                        temperature=1.5,
                                        max_tokens=None,
                                        timeout=None,
                                        max_retries=2,
                                        
                    
                                    )
        human_messaged_mixture = f"""
            This is the list of tags: {asa} and if it the tags list is None Based on the theme asjust the tags to get the nice attractive image in that tags mention don't add the text and the logo and specify the objects will be placed in the right side of banner and the theme is {optional_prompt} and include the objects in the tags it self based on the theme 

            Based on this, please analyze and generate meaningful structured tags. 

            Return the output strictly in the following format:

            {{
            "Tag Name": [],
            "Tag Property": [],
            "Tag Values": []
            }}

            Make sure the output is a valid JSON object and all arrays are aligned by index and in the Colors tags please give me the colors with percentage. and make sure the tag names has equal no of tags amd tag properties have equal no of properties and tag values has equal no of values like len(tag names) = len(tag properties) - len(tag values)
            """
                                
        prompt_for_tags_mixture = [
                        (
                            "system",
                            """Let's Assume you are the best tags analyzer and tags merger based on the high performance and now you want to give me the best outperforming banner tags and based on that tags give me the more tags to improve the image and improve user engagement and don't include the types like doodling and animated etccmand please give me the analyzed tags should be very perfect  and the all tag names and tag properties and tag values will be equal"""
                        ),
                        (
                            "human",
                                human_messaged_mixture
                        )
                        ] 


        
        llm = ChatGoogleGenerativeAI(
                                    model="gemini-2.0-flash",
                                    temperature=1.6,
                                    max_tokens=None,
                                    timeout=None,
                                    max_retries=2,
                
                                )
        resp_tag_mixture = llm.invoke(prompt_for_tags_mixture)
        main_tag_content = str(resp_tag_mixture.content)
        find_idx = main_tag_content.find("{")
        find_idx_l = main_tag_content.rfind("}")
        json_dat_for_tag_mix = main_tag_content[find_idx:find_idx_l+1]
        
        if "the_df_json" not in st.session_state:
            st.session_state["the_df_json"] = json_dat_for_tag_mix
      else:
       st.warning("Please enter your Google API key to continue.")


# Validate that all lists have the same length
  generated_ban_image = None
  



  st.divider()
  st.subheader("Banner Prompt")
  



  
  prompt_input_banner = st.text_input("Enter your banner prompt")

  number_of_images = st.number_input("Enter no of images you want to generate",min_value=2,max_value=5)
  styles = st.selectbox("Select Your Style",["Animated","Modern","Doddling","Typography Animation"])

  if "the_df_json" in st.session_state and styles and prompt_input_banner and number_of_images:
     
     generated_ban_image = st.button("Generate")

  st.divider()



          

  st.subheader("Add TextOverlay to the Image")




with d_rig:
  main_jsoon = None 
  st.subheader("Preview")
  st.text("Based on the Previous Dataset:")
  if biz_sel:
    main_d = st.session_state.get("main_d")
    if main_d is not None and not main_d.empty:
     with st.expander("Expand for Previous Data"):
        st.dataframe(main_d)
    else:
        st.write("No Data Found")
  if "the_df_json" in st.session_state:
         main_jsoon = json.loads(st.session_state["the_df_json"])
         tag_names = main_jsoon.get("Tag Name", [])
         tag_properties = main_jsoon.get("Tag Property", [])
         tag_values = main_jsoon.get("Tag Values", [])
  else:
       st.badge("Not Genrated The Tags")
  if main_jsoon:
   if len(tag_names) == len(tag_properties) == len(tag_values):
    main_tag_mixture_df = pd.DataFrame({
        "Tag Name": tag_names,
        "Tag Property": tag_properties,
        "Tag Values": tag_values
    })
 
    with st.expander("Click for Analyzed tags based on previous Outperforming Banners"):
        st.dataframe(main_tag_mixture_df)
   else:
    st.error(
        f""" 
   Please ensure the AI response returns equal-length lists for all fields.
   """
    )
  else:
       st.warning("Not generated")

      
       
  if generated_ban_image:
    
   st.session_state.img_btn_clicked = True
   if st.session_state.img_btn_clicked and not st.session_state.img_already_generated:
    main_json_file = st.session_state.main_json_file
    json_c = None
    if API_KEY:
        human_message = (
            "Create a promotional banner for the given financial theme. "
            "If people are present, render them fully with correct anatomy, realistic facial expressions, and professional posture. "
            "Objects and people should be positioned on the **right side** of the banner to leave **ample empty space on the top-left** for branding. "
            "Ensure the visual layout is clean, modern, and engaging. Avoid any symbolic or abstract elements unless clearly tied to the theme. "
            "Do not include any text or logos in the image. "
            "Theme: " + prompt_input_banner + " | Style: " + styles +
            ". If the style includes 'animated' or 'cartoon', reflect that in the image accordingly. "
            "Avoid visual metaphors like forests, dreamy fog, glowing lights, or fantasy skies unless directly requested. "
            "The image should look grounded, contextually relevant, and visually aligned with professional Indian financial service advertising."
        )

        prompt = [
            (
                "system",
                f"""
        You are a top-tier promotional-banner generator for Indian financial services campaigns, crafting prompts for Flux 1 Pro image generation.

        == VISUAL RULES ==
        • Do not include text or logos — leave the **top-left** corner clear for future branding.
        • Ensure all main people and objects appear on the **right side** of the image and all the objects will place in the right side only please free space on the left side.
        • Use clean composition, modern spacing, and soft, natural lighting (not fantasy-themed).
        • Avoid clutter, overuse of background fillers, or elements not described in the prompt.
        ensure the bacground will be there in the entire picture but the objects are placed in the right side thats it and don't vanish the background at left and don't give me the white color at all and people will also be placed in the right side only 

        == PEOPLE & STYLING ==
        • Use realistic poses and facial expressions — professional, trustworthy, culturally aligned.
        • Men should wear formal light-colored shirts and trousers (no ethnic or religious clothing).
        • Women should not wear spiritual attire.
        • Maintain natural separation between people and objects to avoid overlaps or distortions.

        == OBJECT PLACEMENT ==
        • Include only those props or visual metaphors that are explicitly relevant to the theme or derived from tags.
        • Do not include  forests, or glowing dreamlike backgrounds unless specifically mentioned.

        == TAG GUIDANCE ==
        Use these tags for reference. If empty or incomplete, infer or create appropriate tags from the theme and style:

        {st.session_state.the_df_json}

        == OUTPUT FORMAT ==
        Return one single flat prompt string ready for Flux 1 Pro — no backticks, no labels, no formatting hints.
        """
            ),
            (
                "human",
                human_message
            )
        ]


        llm = ChatGoogleGenerativeAI(
                                    model="gemini-2.5-pro",
                                    temperature=1.7,
                                    max_tokens=None,
                                    timeout=None,
                                    max_retries=2,
                
                                )


        ai_response = llm.invoke(prompt)
                
        json_c = ai_response.content
        st.session_state['json_c'] = json_c
    else:
        st.warning("Please enter your Google API key to continue.")
    with st.expander("Click Here to View The Generated Prompt"):
       st.success("Generated The image Prompt")
       st.text_area("Generated Prompt:", json_c, height=200)
    st.divider()
    if 'json_c' in st.session_state:
        
        promptt = st.session_state.get("json_c")
        sharedurls = []
        def image_generation_using_flux_1_pro(sharedurls, promptt):
            try:
                randomgen = np.random.randint(1, 100000)
                
                resp = requests.get(
                    f"https://nihalgazi-optimflux.hf.space/?prompt={promptt}&width=1280&height=720&seed={randomgen}",
                    timeout=30
                )
                
               
                if resp.status_code == 200:
                    sharedurls.append(resp.content)
                else:
                    st.error(f"Image Generation Failed: Received status code {resp.status_code}")
            
         
            except requests.exceptions.Timeout:
                st.error("Request timed out. Please try again later.")
            except requests.exceptions.RequestException as e:
                st.error(f"An error occurred while generating the image: {e}")
          
            return sharedurls
        
        try:
            with st.spinner("Generating image..."):
             if number_of_images:
              threadings = []
              for t in range(number_of_images):
                 threadings.append(threading.Thread(target=image_generation_using_flux_1_pro,args=(sharedurls,promptt,)))
              for t in threadings:
                 t.start()
              for t in threadings:
                 t.join()
              if "shared_urls" not in st.session_state:
                     st.session_state["shared_urls"] = sharedurls
        except TimeoutError as e:
            st.error({e})
with d_rig:
    sharedurls = st.session_state.get("shared_urls")
    if sharedurls:
        selected = []
        for t in range(len(sharedurls)):
            if sharedurls[t]:  # Check if sharedurls[t] is not None or empty
                with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmppfile:
                    tmppfile.write(sharedurls[t])
                    file_name = tmppfile.name

                with st.container(border=True):
                    left_col, right_col = st.columns([1, 6])
                    with left_col:
                        checked = st.checkbox(
                            "Select Image", key=f"Image_select_{t}", value=False, label_visibility="collapsed"
                        )

                    with right_col:
                        st.image(file_name)

                    if checked:
                        selected.append(sharedurls[t])
                        
       
        
        if "selected_records" not in st.session_state:
            st.session_state["selected_records"] = selected
        else:
            st.session_state["selected_records"] = selected
            
        if "length_total_records" in st.session_state:
            st.session_state["length_total_records"] = len(selected)
        else:
            st.session_state["length_total_records"] = len(selected)

    

                  
with d_lef:
        text_overlay_prompt = st.text_input("Enter the text prompt you want to add in the image:")
        generate_text_overlay_content_btn = st.button("Generate Content")
        selected = st.session_state.get("selected_records")
        
        if not selected:
            st.warning("No images selected. Please select at least one image.")
        if selected and generate_text_overlay_content_btn and text_overlay_prompt:
            len_selected = st.session_state.get("length_total_records", 0)
            if len_selected >= 1:
                main_file = pd.read_csv("banner_data.csv", encoding='latin-1')
                main_file = main_file.fillna('')

                descriptions = main_file["Title"] + " " + main_file["Description1"] + " " + main_file["Description2"] + " " + main_file["Description3"]
                banner_type = main_file["section"] + " " + main_file["subDivisionType"]
                main_text = banner_type + " " + descriptions

                main_text_ll = [str(text).strip() for text in main_text.to_list()]
                banner_type_ll = [str(bt).strip() for bt in banner_type.to_list()]

                model = SentenceTransformer('all-MiniLM-L6-v2')
                embedding_vector = model.encode(main_text_ll)


               


                try:
                    client = chromadb.PersistentClient(path="./chroma_db")
                    print("Using local ChromaDB")
                except:
                    # Option 2: Cloud client with updated configuration
                    client = chromadb.CloudClient(
                        tenant="4a4205c1-adf8-4989-8824-39983b450e41",
                        database="default",
                        api_key="ck-78tRNWSAkwpJ6szDmFFwkehi46q2FZb2v4Unz3"
                    )
                    print("Using ChromaDB Cloud")


                collection = client.get_or_create_collection(name="Banner_recomm")


                existing_docs = collection.get(ids=None, include=["documents"])["documents"]
                print(f"Total existing docs: {len(existing_docs)}")

                ids_to_add = []
                docs_to_add = []
                embs_to_add = []
                metas_to_add = []
                 
                for i in range(len(main_text_ll)):
                    doc = main_text_ll[i]
                    if doc not in existing_docs: 
                        new_id = str(np.random.randint(1, 10000))
                        while new_id in collection.get(ids=None, include=[])["ids"] or new_id in ids_to_add:
                            new_id = str(np.random.randint(1, 10000))
                        ids_to_add.append(new_id)
                        docs_to_add.append(doc)
                        embs_to_add.append(embedding_vector[i].tolist())
                        metas_to_add.append({"banner_type": banner_type_ll[i]})

                if ids_to_add:
                    print(f"Adding {len(ids_to_add)} new documents...")
                    collection.add(
                        ids=ids_to_add,
                        documents=docs_to_add,
                        embeddings=embs_to_add,
                        metadatas=metas_to_add
                    )
                    print(" Embeddings and documents added successfully!")
                else:
                    print(" All documents already exist.")

                brief = text_overlay_prompt

                qvec  = model.encode(brief).tolist()

                top_k   = 5                     


                qvec = model.encode(brief, convert_to_numpy=True).tolist()
                res  = collection.query(
                    query_embeddings=[qvec],  
                    n_results=top_k,
                    include=["documents", "distances", "metadatas"] 
                )


                ids        = res["ids"][0]         
                texts      = res["documents"][0]   
                distances  = res["distances"][0]   

                for i, (bid, txt, d) in enumerate(zip(ids, texts, distances), 1):
                    print(f"{i}. {bid}  (distance {d:.4f})\n   {txt}\n")
                if API_KEY:

                    human_message = "based on the data given following and the prompt please give me the 3 options for banner title and descriptions like title1 and description1 ,title2 etcc in ajson way don't miss anything give me the complete information.follow the output as title1 and description1 in json etcc. IMPORTANT: Return JSON in this EXACT format: {\"title1\":\"your title\", \"description1\":\"your description\", \"title2\":\"your title\", \"description2\":\"your description\", \"title3\":\"your title\", \"description3\":\"your description\"}. No nested objects, just direct key-value pairs." + str(texts) + " " + brief

                    prompt = [
                        (
                            "system",
                            f"""You are a great banner content creator so based on the past data please give me the engaging and the awesome content for it just give me the json way don't respond with questions with text the reponse will generate the json file and make sure give only title and description please keep the ddescription content short thats it. 

                            CRITICAL: Return JSON in this EXACT format only:
                            {{"title1": "your engaging title", "description1": "short description", "title2": "your engaging title", "description2": "short description", "title3": "your engaging title", "description3": "short description"}}
                            
                            Do NOT use nested objects or arrays. Use direct key-value pairs only."""
                        ),
                        (
                            "human",
                            human_message
                        )
                    ]

                    
                    llm = ChatGoogleGenerativeAI(
                        model="gemini-2.0-flash-lite",
                        temperature=1.5,
                        max_tokens=None,
                        timeout=None,
                        max_retries=2,
                        model_kwargs={"response_format": {"type": "json_object"}}
                    )

                    ai_response = llm.invoke(prompt)
                    recomm_descrip = ai_response.content
                    the_ind = str(ai_response.content)
                    strat = the_ind.find("{")
                    ended = the_ind.rfind("}")
                    main_ye = the_ind[strat:ended+1]

                    print(f"Trying to parse: {main_ye}")
                    

                    try:
                        the_main_ye_json = json.loads(main_ye.strip())
                        print(the_main_ye_json)
                        if "the_main_ye_json" not in st.session_state:
                          st.session_state["the_main_ye_json"] = the_main_ye_json
                    except json.JSONDecodeError as e:
                        print(f"JSON Parse Error: {e}")
                else:
                    st.warning("Please enter your Google API key to continue.")


with d_lef:

    data = st.session_state.get("the_main_ye_json")
    if data:
        ll_len = len(data)

        with st.expander("Expand to select the banner content", expanded=True):
            if ll_len > 1:
            
                    with st.container(border=True):
                       
                        select_option = st.radio(label=" ",options=[f"{i+1} " + data[f"title{i+1}"] + " "+" \n" + data[f"description{i+1}"] + "\n" for i in range(3)])  
                        if "sel_opt" not in st.session_state:
                            st.session_state["sel_opt"] = select_option  
        generate_text_on_the_image = st.button("Generate Text Based Banner")           
                

with d_rig:
    gen_total = []
    length_total_rec = st.session_state.get("length_total_records")
    select_option = st.session_state.get("sel_opt")

    if select_option and generate_text_on_the_image:
        if length_total_rec >= 1:
            for i, image_ in enumerate(selected):
                with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as ttmpfile:
                    ttmpfile.write(image_)
                    selected_file_name = ttmpfile.name

                image = Image.open(selected_file_name).convert("RGBA")
                client = Client("ameerazam08/Gemini-Image-Edit")

                str_option = str(select_option)
                main_idx_of_option = str_option.split()
                idsv = int(main_idx_of_option[0])
                data_text = st.session_state.get("the_main_ye_json")
                main_title = data_text[f"title{idsv}"]
                main_description = data_text[f"description{idsv}"]

               
                for attempt in range(2): 
                    try:
                        result = client.predict(
                            composite_pil=handle_file(selected_file_name),
                            prompt=f"""Add "{main_title}" and "{main_description}" in Rubik bold font, white/black high contrast color. Place LEFT SIDE in free space or if there is no free space in the left side place in the avaliable free space area and don't change the original image at all, NO OVERLAP with objects. Professional banner style, large readable text, proper spacing.""",
                            gemini_api_key=API_KEY,
                            api_name="/process_image_and_prompt"
                        )
                        break  
                    except Exception as e:
                        error_msg = str(e)
                        if "RESOURCE_EXHAUSTED" in error_msg or "429" in error_msg:
                            st.warning(f"Quota exhausted. Waiting 60 seconds before retrying... (Attempt {attempt+1}/2)")
                            time.sleep(60)
                        else:
                            st.error(f"Unexpected error: {e}")
                            result = None
                            break

                if not result:
                    continue

                image_bytes = result[0][0]['image']
                if image_bytes:
                    image = Image.open(image_bytes).convert("RGBA")
                    image = ImageOps.pad(image, (1280, 720), color=(0, 0, 0), centering=(0.5, 0.5))

                    logo_path = "BAJAJFINSV.NS_BIG.png"
                    image_width, image_height = image.size

                    with open(logo_path, "rb") as file:
                        logo = BytesIO(file.read())

                    logo_img = Image.open(logo).convert("RGBA")
                    aspect_ratio = (logo_img.height / logo_img.width)
                    logo_new_width = 150
                    logo_new_height = int(logo_new_width * aspect_ratio)
                    logo_img = logo_img.resize((logo_new_width, logo_new_height), Image.Resampling.LANCZOS)

                    padding_x = 20
                    padding_y = 20
                    position = (image_width - logo_img.width - padding_x, padding_y)
                    image.paste(logo_img, position, mask=logo_img)

                    gen_total.append(image)

        st.session_state["image_total_list"] = gen_total


with d_rig:
            if "image_total_list" in st.session_state:
                        total_ll = st.session_state["image_total_list"]
                        for i,image in enumerate(total_ll):
                            
                         st.image(image,caption="Generated Based On Previous Tags",)
                         if hasattr(image, 'save'):  
   
                            img_buffer = io.BytesIO()
                            image.save(img_buffer, format='PNG')
                            img_buffer.seek(0)
                         else:
                            
                            st.error("Image format not recognized!")

                        # Provide the download button
                         st.download_button(
                            label="Download Image",
                            data=img_buffer,
                            file_name="generated_image.png",
                            mime="image/png",key= f"download_button_{i}"
                         )
                         
                        st.session_state.img_already_generated = True



    




    

            
                
with d_lef:

    if st.button("Reset"):
      st.toast("Click Twice to Reset the Content")
      st.session_state.img_btn_clicked = False
      st.session_state.img_already_generated = False
      st.session_state.text_image_generated = False
      st.session_state.v_tags_btn = False

      keys_to_clear = [
          "image_path",
          "json_c",
          "banner_image",
          "main_json_file",
          "the_main_ye_json",
          "sel_idx",
          "sel_opt",
          "length_total_records",
          "shared_urls",
          "image_total_list",
          "selected_records"
          ""
      ]
      
      for key in keys_to_clear:
          if key in st.session_state:
              del st.session_state[key]