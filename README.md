# Recomendador Musical - Integração MPD/Cantata

Este é um sistema de recomendação musical acústica baseado em IA, que utiliza ChromaDB e HuggingFace (Modelos CLAP) para analisar e recomendar faixas similares ao que você está ouvindo no momento no servidor MPD.

## Como funciona
A aplicação roda em segundo plano observando as alterações no estado do seu player MPD. Quando uma faixa está tocando (ou ao pular de faixa), a IA processa o *fingerprint* sonoro e busca no banco de vetores as faixas mais semelhantes (sem usar metadados de texto), enfileirando automaticamente a melhor recomendação para tocar em seguida!

## Estrutura do Projeto
- `audio_pipeline/`: Código principal em Python (indexação, embeddings e integração com MPD).
- `docs/`: Documentação técnica detalhada da arquitetura, pipeline de áudio, identificação de faixas, recomendação, indexação e avaliação do TCC.
- `mpd.conf.example`: Exemplo de configuração do servidor MPD no Windows.
- `iniciar_sistema.bat`: Script prático para rodar tanto o servidor MPD quanto a IA integradora.

## Documentação Técnica (TCC)
Consulte a documentação completa em `docs/`:
- [docs/arquitetura.md](docs/arquitetura.md): Fluxo geral do sistema (Cantata → MPD → ChromaDB) e responsabilidade de cada módulo.
- [docs/pipeline_embeddings.md](docs/pipeline_embeddings.md): Ingestão, segmentação, modelo CLAP e pooling vetorial.
- [docs/identificacao_faixas.md](docs/identificacao_faixas.md): Normalização de URLs do Cantata, mapeamento de arquivos e coexistência de IDs.
- [docs/recomendacao_e_fila.md](docs/recomendacao_e_fila.md): Eventos do player, histórico de sessão, filtros de candidatas e reposição de fila.
- [docs/indexacao.md](docs/indexacao.md): Comparação entre indexação em lote e indexação sob demanda (deduplicação por fingerprint).
- [docs/avaliacao.md](docs/avaliacao.md): Métrica Precision@K, baseline aleatória e formulação matemática.
- [docs/limitacoes.md](docs/limitacoes.md): Registro explícito de limitações conhecidas e inconsistências arquiteturais.

## Instalação e Uso

1. **Configuração do MPD**
   - Instale o MPD (Music Player Daemon) para Windows.
   - Renomeie o `mpd.conf.example` para `mpd.conf` e edite o `music_directory` para apontar para sua biblioteca de áudio local (ex: `"G:/Meu Drive/Musicas"`).

2. **Ambiente Python**
   ```bash
   cd audio_pipeline
   python -m venv venv
   venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Iniciando**
   - Utilize o script `iniciar_sistema.bat` na raiz do projeto para rodar tudo automaticamente de forma sequencial.

## Painel de Acompanhamento de Métricas (Streamlit)

O painel exibe em tempo real as métricas de Precision@K, histogramas e histórico de execuções.

### Abrir o Painel

| Método | Como fazer |
|--------|-----------|
| **Duplo clique** | `audio_pipeline\results\abrir_painel.bat` |
| **Automático** | O `iniciar_sistema.bat` já sobe o painel (se `ABRIR_PAINEL=1`) |
| **Atalho na área de trabalho** | Execute `audio_pipeline\results\criar_atalho_desktop.ps1` uma vez |

O painel estará disponível em: **http://localhost:8501**

### Parar o Painel

- Execute `audio_pipeline\results\parar_painel.bat` (duplo clique)
- Ou encerre o processo `streamlit.exe` pelo Gerenciador de Tarefas

### Desativar o painel no iniciar_sistema.bat

Edite `iniciar_sistema.bat` e mude a linha:
```bat
set ABRIR_PAINEL=0
```

### Gerar Métricas Manualmente

Dentro do painel, use o botão **"Executar geração"** na barra lateral.
Ou via terminal (dentro de `audio_pipeline\`):
```bash
.\venv\Scripts\python.exe results\generate_precision_k.py --k 5
```
