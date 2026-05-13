from __future__ import annotations

import argparse
import random
from pathlib import Path

import pandas as pd

from src.config import PROJECT_DIR, resolve_project_path

RANDOM_STATE = 42

BASE_EXAMPLES: dict[str, list[str]] = {
    "payment_issue": [
        "списали деньги два раза",
        "деньги списали но заказ не оформился",
        "не проходит оплата картой",
        "ошибка оплаты при оформлении заказа",
        "платеж не прошел",
        "карта не принимается",
        "оплата зависла",
        "списали неправильную сумму",
        "не могу оплатить заказ",
        "после оплаты ничего не изменилось",
        "почему сняли деньги с карты",
        "не пришел чек после оплаты",
        "деньги ушли а подписка не активна",
        "банк подтвердил оплату но заказ не появился",
        "аплата не проходит",
        "списали денги два раза",
    ],
    "refund_request": [
        "хочу вернуть деньги",
        "оформите возврат средств",
        "как сделать возврат",
        "хочу отменить заказ и вернуть деньги",
        "товар не подошел нужен возврат",
        "верните деньги за заказ",
        "нужно вернуть товар",
        "можно ли оформить возврат",
        "хочу отказаться от покупки",
        "возврат денег на карту",
        "сделайте возврат оплаты",
        "верните оплату пожалуйста",
        "мне нужен возврат средств",
        "хачу вернуть деньги",
        "хочу вирнуть деньги",
        "верните денги за заказ",
    ],
    "delivery_issue": [
        "заказ не пришел",
        "доставка задерживается",
        "курьер не приехал",
        "где мой заказ",
        "посылка потерялась",
        "не пришел товар",
        "заказ привезли не туда",
        "нет информации по доставке",
        "доставка стоит на месте",
        "заказ долго не отправляют",
        "курьер не звонит",
        "товар не доставили",
        "почему доставка перенесена",
        "заказ ни пришел",
        "курер не приехал",
        "пасылка потерялась",
    ],
    "account_access": [
        "не могу войти в аккаунт",
        "забыл пароль",
        "аккаунт заблокирован",
        "не приходит код для входа",
        "нужно восстановить доступ",
        "как сменить пароль",
        "пишет неверный логин",
        "не получается авторизоваться",
        "не могу зайти в личный кабинет",
        "код подтверждения не приходит",
        "почта не подходит для входа",
        "помогите восстановить аккаунт",
        "потерял доступ к профилю",
        "ни магу войти в аккаунт",
        "забыл пороль",
        "акккаунт заблокирован",
    ],
    "technical_bug": [
        "ошибка на сайте",
        "кнопка не работает",
        "приложение вылетает",
        "страница не открывается",
        "ошибка сервера",
        "личный кабинет зависает",
        "не работает форма заказа",
        "сайт показывает ошибку",
        "не открывается корзина",
        "приложение зависло",
        "после нажатия ничего не происходит",
        "не загружается страница оплаты",
        "появляется неизвестная ошибка",
        "ашибка на сайте",
        "кнопка ни работает",
        "приложение вылетаит",
    ],
    "general_question": [
        "как изменить данные профиля",
        "где посмотреть тариф",
        "как связаться с поддержкой",
        "какие есть способы оплаты",
        "как поменять адрес доставки",
        "где найти чек",
        "как изменить номер телефона",
        "сколько стоит доставка",
        "как работает подписка",
        "где посмотреть историю заказов",
        "как удалить аккаунт",
        "какие условия доставки",
        "можно ли изменить email",
        "где пасмотреть тариф",
        "как поменять номер телифона",
        "какие способы аплаты есть",
    ],
}

PREFIXES = [
    "",
    "здравствуйте",
    "добрый день",
    "подскажите пожалуйста",
    "помогите пожалуйста",
    "у меня проблема",
    "срочно помогите",
]

SUFFIXES = [
    "",
    "что делать",
    "помогите",
    "пожалуйста",
    "не понимаю что делать",
    "можете проверить",
    "нужно решить проблему",
]


def make_dataset(random_state: int = RANDOM_STATE) -> pd.DataFrame:
    """Create a deterministic synthetic dataset for the project."""
    random.seed(random_state)
    rows: list[dict[str, str]] = []

    for category, examples in BASE_EXAMPLES.items():
        for text in examples:
            rows.append({"text": text, "category": category})

            for _ in range(2):
                prefix = random.choice(PREFIXES)
                suffix = random.choice(SUFFIXES)
                parts = [prefix, text, suffix]
                synthetic_text = " ".join(part for part in parts if part).strip()
                rows.append({"text": synthetic_text, "category": category})

    dataset = pd.DataFrame(rows).drop_duplicates().sample(frac=1.0, random_state=random_state)
    return dataset.reset_index(drop=True)


def save_dataset(output_path: str | Path) -> Path:
    path = resolve_project_path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    dataset = make_dataset()
    dataset.to_csv(path, index=False, encoding="utf-8")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic support ticket dataset")
    parser.add_argument("--output", default="data/tickets.csv", help="Output CSV path")
    args = parser.parse_args()

    output_path = save_dataset(args.output)
    dataset = pd.read_csv(output_path)
    print(f"Saved dataset to {output_path}")
    print(f"Rows: {len(dataset)}")
    print(dataset["category"].value_counts().to_string())


if __name__ == "__main__":
    main()
