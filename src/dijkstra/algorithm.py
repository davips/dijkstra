from math import inf


def dijkstra(origem, grafo):
    # Inicialização
    distancia_até_origem = {nó: inf for nó in grafo.keys()}
    distancia_até_origem[origem] = 0
    visitados = set()
    todos = set(distancia_até_origem.keys())

    # Loop principal
    while visitados != todos:

        # Busca nó não visitado com menor distância conhecida até a origem
        nó_atual = None
        menor_distancia = inf
        for nó in grafo:
            if nó not in visitados and distancia_até_origem[nó] < menor_distancia:
                nó_atual = nó
                menor_distancia = distancia_até_origem[nó]
        print(f"Nó com menor distância conhecida até a origem: {nó_atual}, {menor_distancia}")

        # Marca o nó atual como visitado
        visitados.add(nó_atual)

        # Atualiza as distâncias conhecidas dos nós vizinhos até a origem
        for vizinho, peso in grafo[nó_atual].items():
            if distancia_até_origem[nó_atual] + peso < distancia_até_origem[vizinho]:
                distancia_até_origem[vizinho] = distancia_até_origem[nó_atual] + peso
    # Fim WHILE

    # Retorna as distâncias mais curtas a partir da origem
    return distancia_até_origem


# Definindo o grafo com as conexões e custos
grafo = {
    'a': {'b': 6, 'd': 1},
    'b': {'a': 6, 'c': 5, 'e': 2},
    'c': {'b': 5, 'e': 5},
    'd': {'a': 1, 'e': 1},
    'e': {'b': 2, 'c': 5, 'd': 1}
}

# Ponto de partida
origem = 'a'

# Chamando o algoritmo de Dijkstra para encontrar os caminhos mais curtos a partir de A
caminhos_mais_curto = dijkstra(origem, grafo)

# Exibindo os caminhos mais curtos
for (destino, distancia) in caminhos_mais_curto.items():
    print(f"Caminho mais curto de {origem} para {destino}: {distancia}")
