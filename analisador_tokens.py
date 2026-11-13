import tiktoken

def contar_tokens(texto: str, modelo: str = "gpt-4o") -> int:
    """Retorna a quantidade de tokens em um texto para o modelo especificado."""
    try:
        codificador = tiktoken.encoding_for_model(modelo)
    except KeyError:
        codificador = tiktoken.get_encoding("cl100k_base")
    
    tokens = codificador.encode(texto)
    return len(tokens), tokens

def estimar_custo(qtd_tokens: int, preco_por_milhao: float = 2.50) -> float:
    """Calcula o custo estimado com base no preço por milhão de tokens."""
    return (qtd_tokens / 1_000_000) * preco_por_milhao

def executar_analise(texto_exemplo: str, modelo: str = "gpt-4o"):
    total_tokens, lista_tokens = contar_tokens(texto_exemplo, modelo)
    
    # Exemplo de tabela de preços aproximada por milhão de tokens de entrada
    precos_modelos = {
        "gpt-4o": 2.50,
        "gpt-4o-mini": 0.15,
        "gpt-3.5-turbo": 0.50
    }
    
    preco = precos_modelos.get(modelo, 1.00)
    custo_estimado = estimar_custo(total_tokens, preco)

    print(f"=== Relatório de Uso de Tokens ===")
    print(f"Modelo Utilizado: {modelo}")
    print(f"Total de Caracteres: {len(texto_exemplo)}")
    print(f"Total de Tokens: {total_tokens}")
    print(f"Custo Estimado de Entrada: ${custo_estimado:.6f} USD")
    print("-" * 35)

if __name__ == "__main__":
    texto_teste = """
    A inteligência artificial processa dados dividindo-os em pedaços
    chamados de tokens. Isso pode ser uma palavra inteira ou parte dela.
    """
    executar_analise(texto_teste, modelo="gpt-4o")
