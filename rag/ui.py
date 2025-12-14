import streamlit as st
import os
import re
from implementation import init_rag


def process_text_with_sentences(text, rag):
    sentences = rag.split_sentences(text)

    if len(sentences) <= 1:
        return rag.answer_sentence(text)
    else:
        results = []
        for i, sent in enumerate(sentences):
            if len(sent) > 3:
                result = rag.answer_sentence(sent)
                results.append(result)
            else:
                results.append(sent)

        return ". ".join(results) + "."


st.set_page_config(page_title="Умное исправление текста", page_icon="🔍", layout="wide")

st.title("🔍 Умное исправление текста")

if "rag_initialized" not in st.session_state:
    if api_key:
        with st.spinner("🔄 Загружаем систему исправления..."):
            try:
                st.session_state.rag = init_rag(api_key)
                st.session_state.rag_initialized = True
            except Exception as e:
                st.error(f"❌ Ошибка загрузки: {e}")

if st.session_state.get("rag_initialized"):
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Ввод текста")
        user_text = st.text_area(
            "Введите текст (можно несколько предложений):",
            height=200,
            placeholder="Например: Пётр Первый родился в 1703 году. Раьсиво сущемтвоыало в Древнем Риме...",
            key="text_input",
        )

        if st.button("🔍 Проверить весь текст", type="primary"):
            if user_text.strip():
                with st.spinner("Проверяем каждое предложение..."):
                    try:
                        result = process_text_with_sentences(
                            user_text, st.session_state.rag
                        )
                        st.session_state.last_result = result
                    except Exception as e:
                        st.session_state.last_result = f"Ошибка: {str(e)}"
            else:
                st.warning("Введите текст для проверки")

    with col2:
        st.subheader("Результат исправления")
        if st.session_state.get("last_result"):
            st.success(st.session_state.last_result)

            if user_text and st.session_state.last_result:
                original_sentences = st.session_state.rag.split_sentences(user_text)
                corrected_sentences = st.session_state.rag.split_sentences(
                    st.session_state.last_result
                )

                st.divider()
                st.subheader("📊 Детали исправлений")

                for i, (orig, corr) in enumerate(
                    zip(original_sentences, corrected_sentences)
                ):
                    if orig != corr:
                        st.write(f"**Предложение {i+1}:**")
                        st.write(f"Было: `{orig}`")
                        st.write(f"Стало: `{corr}`")
                        st.write("---")
        else:
            st.info("Здесь появится исправленный текст")
