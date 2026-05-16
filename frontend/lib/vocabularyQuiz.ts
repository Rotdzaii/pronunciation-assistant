export type QuizQuestion = {
  id: string;
  word: string;
  prompt: string;
  choices: string[];
  answer: string;
  explanation: string;
};

export const vocabularyQuestions: QuizQuestion[] = [
  {
    id: 'architecture',
    word: 'Architecture',
    prompt: '“Architecture” gần nghĩa nhất với lựa chọn nào?',
    choices: ['Kiến trúc', 'Ẩm thực', 'Lịch trình', 'Thói quen'],
    answer: 'Kiến trúc',
    explanation: 'Architecture là kiến trúc hoặc phong cách thiết kế công trình.',
  },
  {
    id: 'confident',
    word: 'Confident',
    prompt: 'Từ nào diễn tả “tự tin”?',
    choices: ['Confident', 'Confused', 'Careless', 'Crowded'],
    answer: 'Confident',
    explanation: 'Confident dùng khi bạn tin vào khả năng của mình.',
  },
  {
    id: 'pronounce',
    word: 'Pronounce',
    prompt: '“Pronounce” thường dùng trong ngữ cảnh nào?',
    choices: ['Phát âm một từ', 'Viết bài luận', 'Đọc bản đồ', 'Đổi mật khẩu'],
    answer: 'Phát âm một từ',
    explanation: 'Pronounce nghĩa là phát âm, nói ra âm của một từ.',
  },
];
