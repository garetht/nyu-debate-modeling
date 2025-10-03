from dataclasses import dataclass
from typing import Any, List, TypeVar, Callable, Type, cast


T = TypeVar("T")


def from_str(x: Any) -> str:
    assert isinstance(x, str)
    return x


def from_int(x: Any) -> int:
    assert isinstance(x, int) and not isinstance(x, bool)
    return x


def from_list(f: Callable[[Any], T], x: Any) -> List[T]:
    assert isinstance(x, list)
    return [f(y) for y in x]


def to_class(c: Type[T], x: Any) -> dict:
    assert isinstance(x, c)
    return cast(Any, x).to_dict()


@dataclass
class Question:
    question: str
    question_unique_id: str
    options: List[str]
    writer_label: int
    gold_label: int
    difficult: int

    @staticmethod
    def from_dict(obj: Any) -> 'Question':
        assert isinstance(obj, dict)
        question = from_str(obj.get("question"))
        question_unique_id = from_str(obj.get("question_unique_id"))
        options = from_list(from_str, obj.get("options"))
        writer_label = from_int(obj.get("writer_label"))
        gold_label = from_int(obj.get("gold_label"))
        difficult = from_int(obj.get("difficult"))
        return Question(question, question_unique_id, options, writer_label, gold_label, difficult)

    def to_dict(self) -> dict:
        result: dict = {}
        result["question"] = from_str(self.question)
        result["question_unique_id"] = from_str(self.question_unique_id)
        result["options"] = from_list(from_str, self.options)
        result["writer_label"] = from_int(self.writer_label)
        result["gold_label"] = from_int(self.gold_label)
        result["difficult"] = from_int(self.difficult)
        return result


@dataclass
class QualityTranscript:
    article_id: int
    set_unique_id: str
    batch_num: int
    writer_id: int
    source: str
    title: str
    author: str
    topic: str
    article: str
    questions: List[Question]
    url: str
    license: str

    @staticmethod
    def from_dict(obj: Any) -> 'QualityTranscript':
        assert isinstance(obj, dict)
        article_id = int(from_str(obj.get("article_id")))
        set_unique_id = from_str(obj.get("set_unique_id"))
        batch_num = int(from_str(obj.get("batch_num")))
        writer_id = int(from_str(obj.get("writer_id")))
        source = from_str(obj.get("source"))
        title = from_str(obj.get("title"))
        author = from_str(obj.get("author"))
        topic = from_str(obj.get("topic"))
        article = from_str(obj.get("article"))
        questions = from_list(Question.from_dict, obj.get("questions"))
        url = from_str(obj.get("url"))
        license = from_str(obj.get("license"))
        return QualityTranscript(article_id, set_unique_id, batch_num, writer_id, source, title, author, topic, article, questions, url, license)

    def to_dict(self) -> dict:
        result: dict = {}
        result["article_id"] = from_str(str(self.article_id))
        result["set_unique_id"] = from_str(self.set_unique_id)
        result["batch_num"] = from_str(str(self.batch_num))
        result["writer_id"] = from_str(str(self.writer_id))
        result["source"] = from_str(self.source)
        result["title"] = from_str(self.title)
        result["author"] = from_str(self.author)
        result["topic"] = from_str(self.topic)
        result["article"] = from_str(self.article)
        result["questions"] = from_list(lambda x: to_class(Question, x), self.questions)
        result["url"] = from_str(self.url)
        result["license"] = from_str(self.license)
        return result


def quality_transcript_from_dict(s: Any) -> QualityTranscript:
    return QualityTranscript.from_dict(s)


def quality_transcript_to_dict(x: QualityTranscript) -> Any:
    return to_class(QualityTranscript, x)
