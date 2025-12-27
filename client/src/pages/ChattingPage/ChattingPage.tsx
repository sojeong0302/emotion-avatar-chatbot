import Chat from "../../components/Chat/Chat";

export default function ChattingPage() {
  return (
    <div className="bg-gray-100 p-10 flex gap-6">
      <div className="w-64">캐릭터 부분</div>
      <Chat title="Emotion Avatar Chatbot" />
    </div>
  );
}
