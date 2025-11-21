def ask(prompt):
 """Asks user a yes or no question and returns the answer."""
 while True:
    a = input(f"{prompt} (sim/não): ").strip().lower()
    if a in {"sim","s"}: return "sim"
    if a in {"nao","não","n"}: return "nao"
    print("Use 'sim' ou 'não'.")

def main():
    perguntas = [
    "Gosta de trabalhar em equipe?",
    "Prefere atividades ao ar livre?",
    "Gosta de lidar com números?",
    "Prefere habilidades artísticas?",
    "Sente-se confortável com tecnologia?",
    "Gosta de resolver problemas complexos?",
    "Prefere atividades que envolvam comunicação?",
    "Interessa-se por cuidar de pessoas?",
    "Gosta de desafios físicos?",
    "Prefere ambientes organizados?"
    ]
    a = [ask(q) for q in perguntas]
    # Exemplo de regra simples:
    if a[0]=="sim" and a[4]=="sim" and a[5]=="sim":
        print("Sugestão: Engenharia/Computação/Ciência de Dados.")
    else:
        print("Sugestão geral: explore o que mais marcou 'sim'.")

if __name__ == "__main__":
    main()