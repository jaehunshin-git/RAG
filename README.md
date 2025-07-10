# RAG Implementation Comparison

This project demonstrates two different approaches to building a Retrieval-Augmented Generation (RAG) system: a custom implementation and a LangChain-based implementation. Both systems use a cafe menu as their knowledge base to answer questions.

## Project Structure

- `config.py`: Configuration file for project-wide settings, such as data file paths.
- `data/`: Contains the knowledge base (`cafe_menu.txt`) and stored vector stores.
- `requirements.txt`: Python dependencies for the project.
- `uv.lock`: Lock file for `uv` package manager.
- `__pycache__/`: Python bytecode cache.
- `1-load/`: Contains scripts for loading documents.
- `2-split/`: Contains scripts for splitting documents into chunks.
- `3-embed/`: Contains scripts for generating embeddings from document chunks.
- `4-store/`: Contains scripts for storing embeddings in a vector store.
- `5-search/`: Contains a custom RAG implementation (`search.py`) and related utilities.
- `6-langchain/`: Contains a LangChain-based RAG implementation (`langchain_ver.py`).

## Features

- **Custom RAG Pipeline:** A step-by-step implementation of load, split, embed, store, and search.
- **LangChain RAG Pipeline:** An equivalent RAG system built using the LangChain framework for comparison.
- **OpenAI Integration:** Utilizes OpenAI's embedding models (`text-embedding-3-small`) and chat models (`gpt-4o-mini`) for embedding generation and answer generation.
- **FAISS Integration (LangChain):** The LangChain version uses FAISS for efficient similarity search.
- **Cafe Menu Knowledge Base:** Demonstrates RAG capabilities using a simple text file as the knowledge source.

## Setup

### Prerequisites

- Python 3.9+
- An OpenAI API Key

### Installation

1.  **Clone the repository:**
    ```bash
    git clone <repository_url>
    cd RAG
    ```
2.  **Install dependencies:**
    It is recommended to use `uv` for package management, but `pip` can also be used.

    Using `uv` (recommended):
    ```bash
    uv pip install -r requirements.txt
    ```
    Using `pip`:
    ```bash
    pip install -r requirements.txt
    ```

3.  **Set up OpenAI API Key:**
    Create a `.env` file in the project root directory and add your OpenAI API key:
    ```
    OPENAI_API_KEY="your_openai_api_key_here"
    ```

## Usage

### Running the Custom RAG Implementation

To run the custom RAG implementation, execute `search.py`:

```bash
python 5-search/search.py
```
This script will perform the load, split, embed, store, and search steps, then generate an answer based on the retrieved information.

### Running the LangChain RAG Implementation

To run the LangChain-based RAG implementation, execute `langchain_ver.py`:

```bash
python 6-langchain/langchain_ver.py
```
This script will also perform the RAG steps using LangChain components and generate an answer.

---

# RAG 구현 비교

이 프로젝트는 검색 증강 생성(RAG) 시스템을 구축하는 두 가지 다른 접근 방식, 즉 사용자 정의 구현과 LangChain 기반 구현을 보여줍니다. 두 시스템 모두 카페 메뉴를 지식 기반으로 사용하여 질문에 답변합니다.

## 프로젝트 구조

- `config.py`: 데이터 파일 경로와 같은 프로젝트 전체 설정을 위한 구성 파일입니다.
- `data/`: 지식 기반(`cafe_menu.txt`) 및 저장된 벡터 저장소를 포함합니다.
- `requirements.txt`: 프로젝트의 Python 종속성입니다.
- `uv.lock`: `uv` 패키지 관리자를 위한 잠금 파일입니다.
- `__pycache__/`: Python 바이트코드 캐시입니다.
- `1-load/`: 문서 로딩을 위한 스크립트를 포함합니다.
- `2-split/`: 문서를 청크로 분할하기 위한 스크립트를 포함합니다.
- `3-embed/`: 문서 청크에서 임베딩을 생성하기 위한 스크립트를 포함합니다.
- `4-store/`: 임베딩을 벡터 저장소에 저장하기 위한 스크립트를 포함합니다.
- `5-search/`: 사용자 정의 RAG 구현(`search.py`) 및 관련 유틸리티를 포함합니다.
- `6-langchain/`: LangChain 기반 RAG 구현(`langchain_ver.py`)을 포함합니다.

## 기능

- **사용자 정의 RAG 파이프라인:** 로드, 분할, 임베딩, 저장 및 검색의 단계별 구현.
- **LangChain RAG 파이프라인:** 비교를 위해 LangChain 프레임워크를 사용하여 구축된 동등한 RAG 시스템.
- **OpenAI 통합:** 임베딩 생성 및 답변 생성을 위해 OpenAI의 임베딩 모델(`text-embedding-3-small`) 및 채팅 모델(`gpt-4o-mini`)을 활용합니다.
- **FAISS 통합 (LangChain):** LangChain 버전은 효율적인 유사성 검색을 위해 FAISS를 사용합니다.
- **카페 메뉴 지식 기반:** 간단한 텍스트 파일을 지식 소스로 사용하여 RAG 기능을 시연합니다.

## 설정

### 전제 조건

- Python 3.9+
- OpenAI API 키

### 설치

1.  **저장소 복제:**
    ```bash
    git clone <repository_url>
    cd RAG
    ```
2.  **종속성 설치:**
    패키지 관리를 위해 `uv`를 사용하는 것이 권장되지만, `pip`도 사용할 수 있습니다.

    `uv` 사용 (권장):
    ```bash
    uv pip install -r requirements.txt
    ```
    `pip` 사용:
    ```bash
    pip install -r requirements.txt
    ```

3.  **OpenAI API 키 설정:**
    프로젝트 루트 디렉토리에 `.env` 파일을 생성하고 OpenAI API 키를 추가합니다:
    ```
    OPENAI_API_KEY="your_openai_api_key_here"
    ```

## 사용법

### 사용자 정의 RAG 구현 실행

사용자 정의 RAG 구현을 실행하려면 `search.py`를 실행합니다:

```bash
python 5-search/search.py
```
이 스크립트는 로드, 분할, 임베딩, 저장 및 검색 단계를 수행한 다음 검색된 정보를 기반으로 답변을 생성합니다.

### LangChain RAG 구현 실행

LangChain 기반 RAG 구현을 실행하려면 `langchain_ver.py`를 실행합니다:

```bash
python 6-langchain/langchain_ver.py
```
이 스크립트 또한 LangChain 구성 요소를 사용하여 RAG 단계를 수행하고 답변을 생성합니다.
